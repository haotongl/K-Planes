import glob
import json
import logging as log
import math
import os
import time
from collections import defaultdict
from typing import Optional, List, Tuple, Any, Dict
from os.path import join
from tqdm import tqdm
import imageio
import cv2

import numpy as np
import torch

from .base_dataset import BaseDataset
from .data_loading import parallel_load_images
from .intrinsics import Intrinsics
from .llff_dataset import load_llff_poses_helper
from .ray_utils import (
    generate_spherical_poses, create_meshgrid, stack_camera_dirs, get_rays, generate_spiral_path
)
from .synthetic_nerf_dataset import (
    load_360_images, load_360_intrinsics,
)


def get_bound_corners(bounds):
    min_x, min_y, min_z = bounds[0]
    max_x, max_y, max_z = bounds[1]
    corners_3d = np.array([
        [min_x, min_y, min_z],
        [min_x, min_y, max_z],
        [min_x, max_y, min_z],
        [min_x, max_y, max_z],
        [max_x, min_y, min_z],
        [max_x, min_y, max_z],
        [max_x, max_y, min_z],
        [max_x, max_y, max_z],
    ])
    return corners_3d


class Video360Dataset(BaseDataset):
    len_time: int
    max_cameras: Optional[int]
    max_tsteps: Optional[int]
    timestamps: Optional[torch.Tensor]

    def __init__(self,
                 datadir: str,
                 split: str,
                 batch_size: Optional[int] = None,
                 downsample: float = 1.0,
                 keyframes: bool = False,
                 max_cameras: Optional[int] = None,
                 max_tsteps: Optional[int] = None,
                 isg: bool = False,
                 contraction: bool = False,
                 ndc: bool = False,
                 scene_bbox: Optional[List] = None,
                 near_scaling: float = 0.9,
                 ndc_far: float = 2.6):
        self.keyframes = keyframes
        self.max_cameras = max_cameras
        self.max_tsteps = max_tsteps
        self.downsample = downsample
        self.isg = isg
        self.ist = False
        # self.lookup_time = False
        self.per_cam_near_fars = None
        self.global_translation = torch.tensor([0, 0, 0])
        self.global_scale = torch.tensor([1, 1, 1])
        self.near_scaling = near_scaling
        self.ndc_far = ndc_far
        self.median_imgs = None
        if contraction and ndc:
            raise ValueError("Options 'contraction' and 'ndc' are exclusive.")
        if "lego" in datadir or "dnerf" in datadir or 'nhr' in datadir:
            dset_type = "synthetic"
        else:
            dset_type = "llff"

        self.split = split
        # Note: timestamps are stored normalized between -1, 1.
        # self.name = 'sport_1_easymocap'
        self.name = datadir.split('/')[-1]
        self.annots = np.load(join(datadir, 'annots_new.npy'), allow_pickle=True).item()
        frame_len = 200
        if split == 'train':
            cams = np.arange(1, len(self.annots['ixts']))
        else:
            cams = np.arange(0, 1)

        H, W = 768, 1024
        rs = 1. / downsample
        H, W = int(H*rs), int(W*rs)
        X, Y = np.meshgrid(np.arange(W), np.arange(H))
        Z = np.ones_like(X)
        XYZ = np.stack([X, Y, Z], axis=-1)
        self.imgs, self.timestamps, self.ray_ds, self.cam_ids, self.near_fars = \
                [], [], [], [], []
        for cam in tqdm(cams):
            ext = self.annots['exts'][cam]
            ixt = self.annots['ixts'][cam].copy()
            if rs != 1.:
                ixt[:2] *= rs
            for frame_id in range(frame_len):
                img = imageio.imread(join(datadir, 'images', '{:04d}'.format(cam), '{:06d}.jpg'.format(frame_id)))
                msk = imageio.imread(join(datadir, 'mask', '{:04d}'.format(cam), '{:06d}.png'.format(frame_id)))
                img[msk==0] = 0.
                img = cv2.resize(img, None, fx=rs, fy=rs, interpolation=cv2.INTER_AREA)
                self.imgs.append(img)
                # self.imgs = np.concatenate([self.imgs, img[None]], axis=0)
                self.timestamps.append(frame_id/frame_len * 2. - 1.)
                self.cam_ids.append(cam)
                # near, far
                points = get_bound_corners(self.annots['bboxs'][frame_id]) @ ext[:3, :3].T + ext[:3, 3:].T
                self.near_fars.append([max(points[..., 2].min(), 0.05), points[..., 2].max()])
                ray_d = XYZ @ np.linalg.inv(ixt.T) @ np.linalg.inv(ext[:3, :3].T)
                self.ray_ds.append(ray_d.astype(np.float32))
                # self.ray_ds = np.concatenate([self.ray_ds, ray_d[None].astype(np.float32)], axis=0)
        self.imgs = np.array(self.imgs)
        self.ray_ds = np.array(self.ray_ds)
        self.cam_ids = np.array(self.cam_ids)
        self.near_fars = np.array(self.near_fars).astype(np.float32)
        self.timestamps = np.array(self.timestamps).astype(np.float32)
        self.is_contracted = False
        self.is_ndc = False
        self.datadir = datadir
        self.poses = np.linalg.inv(self.annots['exts']).astype(np.float32)
        self.img_h_ = H
        self.img_w_ = W
        self.scene_bbox = torch.tensor(np.array(scene_bbox).astype(np.float32))
        self.sampling_weights = None
        self.use_permutation = True
        self.batch_size = batch_size
        self.num_samples = len(self.imgs) * self.batch_size if self.split == 'train' else len(self.imgs)
        log.info(f"VideoDataset contracted={self.is_contracted}, ndc={self.is_ndc}. "
                 f"Loaded {self.split} set from {self.datadir}: ")

    def enable_isg(self):
        self.isg = True
        self.ist = False
        self.sampling_weights = self.isg_weights
        log.info(f"Enabled ISG weights.")

    def switch_isg2ist(self):
        self.isg = False
        self.ist = True
        self.sampling_weights = self.ist_weights
        log.info(f"Switched from ISG to IST weights.")

    def __getitem__(self, index):
        h, w = self.img_h_, self.img_w_
        coords = np.stack(np.meshgrid(np.arange(0, w), np.arange(0, h)), -1)

        if self.split == 'train':
            coords = coords.reshape(-1, 2)
            select_inds = np.random.choice(coords.shape[0], size=[self.batch_size], replace=True)
            coords = coords[select_inds]
            img_ids = np.random.choice(len(self.imgs), size=[self.batch_size], replace=True)
            coords = np.concatenate([coords, img_ids[..., None]], axis=-1)

            rays_d = self.ray_ds[coords[..., 2], coords[..., 1], coords[..., 0]]
            rgb = self.imgs[coords[..., 2], coords[..., 1], coords[..., 0]]
            cam_ids = self.cam_ids[coords[..., 2]]
            rays_o = self.poses[:, :3, 3][cam_ids]
            near_fars = self.near_fars[coords[..., 2]]
            timestamps = self.timestamps[coords[..., 2]]

            out = {'timestamps': timestamps}
            out.update({'imgs': (rgb / 255.).astype(np.float32), 'rays_o': rays_o, 'rays_d': rays_d, 'near_fars': near_fars})
            bg_color = np.zeros((1, 3), dtype=np.float32)
            out.update({'bg_color': bg_color})
        else:
            out = {}
            out['imgs'] = torch.tensor((self.imgs[index].reshape(-1, 3) / 255.).astype(np.float32))
            out['timestamps'] = torch.tensor(self.timestamps[index])
            out['near_fars'] = torch.tensor(self.near_fars[index:index+1])
            out['rays_o'] = torch.tensor(self.poses[index, :3, 3][None].repeat(len(out['imgs']), 0))
            out['rays_d'] = torch.tensor(self.ray_ds[index].reshape(-1, 3))
            bg_color = torch.zeros((1, 3))
            out.update({'bg_color': bg_color})

        return out


def get_bbox(datadir: str, dset_type: str, is_contracted=False) -> torch.Tensor:
    """Returns a default bounding box based on the dataset type, and contraction state.

    Args:
        datadir (str): Directory where data is stored
        dset_type (str): A string defining dataset type (e.g. synthetic, llff)
        is_contracted (bool): Whether the dataset will use contraction

    Returns:
        Tensor: 3x2 bounding box tensor
    """
    if is_contracted:
        radius = 2
    elif dset_type == 'synthetic':
        radius = 1.5
    elif dset_type == 'llff':
        return torch.tensor([[-3.0, -1.67, -1.2], [3.0, 1.67, 1.2]])
    else:
        radius = 1.3
    return torch.tensor([[-radius, -radius, -radius], [radius, radius, radius]])


def fetch_360vid_info(frame: Dict[str, Any]):
    timestamp = None
    fp = frame['file_path']
    if '_r' in fp:
        timestamp = int(fp.split('t')[-1].split('_')[0])
    if 'r_' in fp:
        pose_id = int(fp.split('r_')[-1])
    else:
        pose_id = int(fp.split('r')[-1])
    if timestamp is None:  # will be None for dnerf
        timestamp = frame['time']
    return timestamp, pose_id


def load_360video_frames(datadir, split, max_cameras: int, max_tsteps: Optional[int]) -> Tuple[Any, Any]:
    with open(os.path.join(datadir, f"transforms_{split}.json"), 'r') as fp:
        meta = json.load(fp)
    frames = meta['frames']

    timestamps = set()
    pose_ids = set()
    fpath2poseid = defaultdict(list)
    for frame in frames:
        timestamp, pose_id = fetch_360vid_info(frame)
        timestamps.add(timestamp)
        pose_ids.add(pose_id)
        fpath2poseid[frame['file_path']].append(pose_id)
    timestamps = sorted(timestamps)
    pose_ids = sorted(pose_ids)

    if max_cameras is not None:
        num_poses = min(len(pose_ids), max_cameras or len(pose_ids))
        subsample_poses = int(round(len(pose_ids) / num_poses))
        pose_ids = set(pose_ids[::subsample_poses])
        log.info(f"Selected subset of {len(pose_ids)} camera poses: {pose_ids}.")

    if max_tsteps is not None:
        num_timestamps = min(len(timestamps), max_tsteps or len(timestamps))
        subsample_time = int(math.floor(len(timestamps) / (num_timestamps - 1)))
        timestamps = set(timestamps[::subsample_time])
        log.info(f"Selected subset of timestamps: {sorted(timestamps)} of length {len(timestamps)}")

    sub_frames = []
    for frame in frames:
        timestamp, pose_id = fetch_360vid_info(frame)
        if timestamp in timestamps and pose_id in pose_ids:
            sub_frames.append(frame)
    # We need frames to be sorted by pose_id
    sub_frames = sorted(sub_frames, key=lambda f: fpath2poseid[f['file_path']])
    return sub_frames, meta


def load_llffvideo_poses(datadir: str,
                         downsample: float,
                         split: str,
                         near_scaling: float) -> Tuple[
                            torch.Tensor, torch.Tensor, Intrinsics, List[str]]:
    """Load poses and metadata for LLFF video.

    Args:
        datadir (str): Directory containing the videos and pose information
        downsample (float): How much to downsample videos. The default for LLFF videos is 2.0
        split (str): 'train' or 'test'.
        near_scaling (float): How much to scale the near bound of poses.

    Returns:
        Tensor: A tensor of size [N, 4, 4] containing c2w poses for each camera.
        Tensor: A tensor of size [N, 2] containing near, far bounds for each camera.
        Intrinsics: The camera intrinsics. These are the same for every camera.
        List[str]: List of length N containing the path to each camera's data.
    """
    poses, near_fars, intrinsics = load_llff_poses_helper(datadir, downsample, near_scaling)

    videopaths = np.array(glob.glob(os.path.join(datadir, '*.mp4')))  # [n_cameras]
    assert poses.shape[0] == len(videopaths), \
        'Mismatch between number of cameras and number of poses!'
    videopaths.sort()

    # The first camera is reserved for testing, following https://github.com/facebookresearch/Neural_3D_Video/releases/tag/v1.0
    if split == 'train':
        split_ids = np.arange(1, poses.shape[0])
    elif split == 'test':
        split_ids = np.array([0])
    else:
        split_ids = np.arange(poses.shape[0])
    if 'coffee_martini' in datadir:
        # https://github.com/fengres/mixvoxels/blob/0013e4ad63c80e5f14eb70383e2b073052d07fba/dataLoader/llff_video.py#L323
        log.info(f"Deleting unsynchronized camera from coffee-martini video.")
        split_ids = np.setdiff1d(split_ids, 12)
    poses = torch.from_numpy(poses[split_ids])
    near_fars = torch.from_numpy(near_fars[split_ids])
    videopaths = videopaths[split_ids].tolist()

    return poses, near_fars, intrinsics, videopaths


def load_llffvideo_data(videopaths: List[str],
                        cam_poses: torch.Tensor,
                        intrinsics: Intrinsics,
                        split: str,
                        keyframes: bool,
                        keyframes_take_each: Optional[int] = None,
                        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if keyframes and (keyframes_take_each is None or keyframes_take_each < 1):
        raise ValueError(f"'keyframes_take_each' must be a positive number, "
                         f"but is {keyframes_take_each}.")

    loaded = parallel_load_images(
        dset_type="video",
        tqdm_title=f"Loading {split} data",
        num_images=len(videopaths),
        paths=videopaths,
        poses=cam_poses,
        out_h=intrinsics.height,
        out_w=intrinsics.width,
        load_every=keyframes_take_each if keyframes else 1,
    )
    imgs, poses, median_imgs, timestamps = zip(*loaded)
    # Stack everything together
    timestamps = torch.cat(timestamps, 0)  # [N]
    poses = torch.cat(poses, 0)            # [N, 3, 4]
    imgs = torch.cat(imgs, 0)              # [N, h, w, 3]
    median_imgs = torch.stack(median_imgs, 0)  # [num_cameras, h, w, 3]

    return poses, imgs, timestamps, median_imgs


@torch.no_grad()
def dynerf_isg_weight(imgs, median_imgs, gamma):
    # imgs is [num_cameras * num_frames, h, w, 3]
    # median_imgs is [num_cameras, h, w, 3]
    assert imgs.dtype == torch.uint8
    assert median_imgs.dtype == torch.uint8
    num_cameras, h, w, c = median_imgs.shape
    squarediff = (
        imgs.view(num_cameras, -1, h, w, c)
            .float()  # creates new tensor, so later operations can be in-place
            .div_(255.0)
            .sub_(
                median_imgs[:, None, ...].float().div_(255.0)
            )
            .square_()  # noqa
    )  # [num_cameras, num_frames, h, w, 3]
    # differences = median_imgs[:, None, ...] - imgs.view(num_cameras, -1, h, w, c)  # [num_cameras, num_frames, h, w, 3]
    # squarediff = torch.square_(differences)
    psidiff = squarediff.div_(squarediff + gamma**2)
    psidiff = (1./3) * torch.sum(psidiff, dim=-1)  # [num_cameras, num_frames, h, w]
    return psidiff  # valid probabilities, each in [0, 1]


@torch.no_grad()
def dynerf_ist_weight(imgs, num_cameras, alpha=0.1, frame_shift=25):  # DyNerf uses alpha=0.1
    assert imgs.dtype == torch.uint8
    N, h, w, c = imgs.shape
    frames = imgs.view(num_cameras, -1, h, w, c).float()  # [num_cameras, num_timesteps, h, w, 3]
    max_diff = None
    shifts = list(range(frame_shift + 1))[1:]
    for shift in shifts:
        shift_left = torch.cat([frames[:, shift:, ...], torch.zeros(num_cameras, shift, h, w, c)], dim=1)
        shift_right = torch.cat([torch.zeros(num_cameras, shift, h, w, c), frames[:, :-shift, ...]], dim=1)
        mymax = torch.maximum(torch.abs_(shift_left - frames), torch.abs_(shift_right - frames))
        if max_diff is None:
            max_diff = mymax
        else:
            max_diff = torch.maximum(max_diff, mymax)  # [num_timesteps, h, w, 3]
    max_diff = torch.mean(max_diff, dim=-1)  # [num_timesteps, h, w]
    max_diff = max_diff.clamp_(min=alpha)
    return max_diff
