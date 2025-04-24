# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import os
import numpy as np
import click
import imageio as imageio_v1
import torch
import cv2
import json
import decord

from tqdm import tqdm
from pathlib import Path
from termcolor import cprint
from pathlib import Path
from utils.wds_utils import get_sample
from utils.bbox_utils import create_bbox_projection, interpolate_bbox, fix_static_objects
from utils.minimap_utils import create_minimap_projection, simplify_minimap
from utils.pcd_utils import batch_move_points_within_bboxes_sparse, forward_warp_multiframes_sparse_depth_only
from utils.camera.pinhole import PinholeCamera
from utils.camera.ftheta import FThetaCamera
from utils.multigpu_utils import setup

INPUT_POSE_FPS = 30
INPUT_LIDAR_FPS = 10
CUT_LEN = 121
OVERLAP = 9

def get_low_fps_indices(high_indices, step=3):
    low_indices = []
    low_indices_times_step = []
    for hi in high_indices:
        # Round to the nearest multiple of 'step'
        low_idx = int(round(hi / step))
        low_idx_times_step = low_idx * step
        low_indices.append(low_idx)
        low_indices_times_step.append(low_idx_times_step)
    return low_indices_times_step, low_indices

def render_sample(
        input_root: str,
        output_root: str,
        clip_id: str,
        settings: dict,
        camera_type: str,
        skip: list[str],
        output_folder: str,
        post_training: bool = False,
        target_resolution: tuple[int, int] = (1280, 720),
        crop_resolution: tuple[int, int] = (1280, 704),
        accumulate_lidar_frames: int = 2,
        transfer_format: bool = False,
):
    if post_training: 
        TARGET_RENDER_FPS = 10
        CUT_LEN = 190 # for waymo post-training, we don't cut the video to 121
    else: 
        TARGET_RENDER_FPS = 30
    CAMERAS = settings['CAMERAS']
    minimap_types = settings['minimap_types']

    target_w, target_h = target_resolution
    # generate map projection for the instance buffer
    for camera_name in CAMERAS:
        print(f"Processing {clip_id} {camera_name}...")
        # load pose
        pose_file = os.path.join(input_root, 'pose', f"{clip_id}.tar")
        camera_key = f"pose.{camera_name}.npy"
        pose_data = get_sample(pose_file)
        pose_data_this_cam = {k: v for k, v in pose_data.items() if camera_key in k}
        pose_this_cam_array = np.stack([pose_data_this_cam[k] for k in sorted(pose_data_this_cam.keys())])

        frame_num = pose_this_cam_array.shape[0]
        valid_frame_ids = list(range(0, frame_num, INPUT_POSE_FPS // TARGET_RENDER_FPS))

        # interpolate bbox from 10Hz to 30Hz
        all_object_info_file = os.path.join(input_root, 'all_object_info', f"{clip_id}.tar")
        all_object_info = get_sample(all_object_info_file)

        # fix static objects in jitter
        all_object_info = fix_static_objects(all_object_info)

        # interpolate bbox if box label FPS is lower than TARGET_RENDER_FPS
        all_object_info = interpolate_bbox(all_object_info, valid_frame_ids)

        # read intrinsic
        if camera_type == "pinhole":
            intrinsic_file = os.path.join(input_root, f'{camera_type}_intrinsic', f"{clip_id}.tar")
            intrinsic_data = get_sample(intrinsic_file)
            intrinsic_this_cam = intrinsic_data[f"{camera_type}_intrinsic.{camera_name}.npy"]

            fx, fy, cx, cy, width, height = intrinsic_this_cam.tolist()
            downsample_ratio = width / target_w
            downsample_ratio_h = height / target_h

            fx_downsample = fx / downsample_ratio
            fy_downsample = fy / downsample_ratio_h
            cx_downsample = cx / downsample_ratio
            cy_downsample = cy / downsample_ratio_h
            width_downsample = width / downsample_ratio
            height_downsample = height / downsample_ratio_h

            intrinsic_downsample = np.array([fx_downsample, fy_downsample, cx_downsample, cy_downsample, width_downsample, height_downsample])
            assert intrinsic_downsample.shape[0] == intrinsic_this_cam.shape[0] # fx fy cx cy width height
            intrinsic_this_cam = intrinsic_downsample

        elif camera_type == "ftheta":
            intrinsic_file = os.path.join(input_root, f'{camera_type}_intrinsic', f"{clip_id}.tar")

            # 3-rd party dataset does not have ftheta intrinsic, we can use the default one from RDS-HQ
            if os.path.exists(intrinsic_file):
                intrinsic_data = get_sample(intrinsic_file)
                intrinsic_this_cam = intrinsic_data[f"{camera_type}_intrinsic.{camera_name}.npy"]

            else:
                cprint(f"Ftheta intrinsic file does not exist: {intrinsic_file}", 'red')
                cprint(f"===> So we will use default ftheta intrinsic for rendering", 'yellow', attrs=['bold'])
                intrinsic_file = 'config/default_ftheta_intrinsic.tar'
                camera_name_in_rds_hq = settings['CAMERAS_TO_RDS_HQ'][camera_name]    
                intrinsic_data = get_sample(intrinsic_file)
                intrinsic_this_cam = intrinsic_data[f"{camera_type}_intrinsic.{camera_name_in_rds_hq}.npy"]
            
            cx, cy, width, height, bw_poly, is_bw_poly = intrinsic_this_cam[0], intrinsic_this_cam[1], intrinsic_this_cam[2], intrinsic_this_cam[3], intrinsic_this_cam[4:-1], intrinsic_this_cam[-1]
            if not is_bw_poly > 0:
                # create a camera_model to compute the bw_poly
                camera_model = FThetaCamera.from_numpy(intrinsic_this_cam, device='cpu')
                bw_poly = camera_model._compute_bw_poly()
                is_bw_poly = torch.tensor([1], dtype=torch.float32)
        
            assert is_bw_poly == 1
            downsample_ratio = width / target_w
            downsample_ratio_h = height / target_h
            assert abs(downsample_ratio_h - downsample_ratio) < 0.01 # only handle the case that height is downsampled by the same ratio, required by the ftheta camera model
            bw_poly_downsample = [p * downsample_ratio**i for i, p in enumerate(bw_poly)]
            cx_downsample = cx / downsample_ratio
            cy_downsample = cy / downsample_ratio_h
            height_downsample = height / downsample_ratio_h
            width_downsample = width / downsample_ratio
            intrinsic_downsample = np.concatenate([cx_downsample.reshape([1]), cy_downsample.reshape([1]), width_downsample.reshape([1]), height_downsample.reshape([1]), bw_poly_downsample, is_bw_poly.reshape([1])])
            assert intrinsic_downsample.shape[0] == intrinsic_this_cam.shape[0] # cx cy width height bw_poly is_bw_poly
            intrinsic_this_cam = intrinsic_downsample

        else:
            raise ValueError(f"Invalid camera type: {camera_type}")
        
        if camera_type == 'pinhole':
            camera_model = PinholeCamera.from_numpy(intrinsic_this_cam, device='cpu')
        elif camera_type == 'ftheta':
            camera_model = FThetaCamera.from_numpy(intrinsic_this_cam, device='cpu')
        else:
            raise ValueError(f"Invalid camera type: {camera_type}")
        
        # HD map and bounding box projection
        if 'hdmap' not in skip:
            minimap_wds_files = [
                os.path.join(input_root, f"3d_{minimap_type}", f"{clip_id}.tar") for minimap_type in minimap_types
            ]

            minimaps_projection_merged = np.zeros((len(valid_frame_ids), camera_model.height, camera_model.width, 3), dtype=np.uint8)
            for minimap_wds_file in minimap_wds_files:
                minimap_data_wo_meta_info, minimap_name = simplify_minimap(minimap_wds_file)

                # all static labels
                minimap_projection = create_minimap_projection(
                    minimap_name,
                    minimap_data_wo_meta_info,
                    pose_this_cam_array[valid_frame_ids],
                    camera_model
                )
                minimaps_projection_merged = np.maximum(minimaps_projection_merged, minimap_projection)

            # add bounding box projection to the minimap
            bounding_box_projection = create_bbox_projection(
                all_object_info,
                pose_this_cam_array,
                valid_frame_ids,
                camera_model,
            )
            minimaps_projection_merged = np.maximum(minimaps_projection_merged, bounding_box_projection)

            # resize minimaps_projection_merged to target_resolution
            # check if the target_resolution is the same as the camera resolution
            if (not target_h == minimaps_projection_merged.shape[1]) or (not target_w == minimaps_projection_merged.shape[2]):
                print(f"Resize minimaps_projection_merged from {minimaps_projection_merged.shape} to {target_resolution}")
                minimaps_projection_merged = [cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4) for frame in minimaps_projection_merged]
                minimaps_projection_merged = np.stack(minimaps_projection_merged, axis=0) # T, H, W, 3

        # LiDAR projection
        if 'lidar' not in skip:
            # load lidar data
            lidar_tar = os.path.join(input_root, 'lidar_raw', f"{clip_id}.tar")
            lidar_data = get_sample(lidar_tar) # 'xxxxxx.xyz' & 'xxxxxx.lidar_to_world'

            # generate lidar depth
            lidar_depth_list = []
            interp_rate = INPUT_POSE_FPS // INPUT_LIDAR_FPS
            for frame_idx in tqdm(valid_frame_ids):
                accumulate_idx = [min(max(0, frame_idx + i * interp_rate), frame_num-1) for i in range(-accumulate_lidar_frames, accumulate_lidar_frames+1)]
                low_fps_frame_indices, _ = get_low_fps_indices(accumulate_idx, step=interp_rate)

                frame_ids = [
                    int(key.split('.')[0])
                    for key in lidar_data.keys()
                    if key.endswith('.lidar_raw.npz') and key.split('.')[0].isdigit()
                ]
                max_lidar_idx = max(frame_ids)
                low_fps_frame_indices = [min(max_lidar_idx, idx) for idx in low_fps_frame_indices]

                accumulated_points, original_bbox_dicts, moved_bbox_dicts = [], [], []

                for i in range(len(accumulate_idx)):
                    lidar_points = lidar_data[f'{low_fps_frame_indices[i]:06d}.lidar_raw.npz']['xyz'].astype(np.float32).reshape(-1, 3)

                    distances = np.linalg.norm(lidar_points, axis=1)
                    threshold = 3.0  
                    lidar_points = lidar_points[distances >= threshold]
                    lidar_to_world = lidar_data[f'{low_fps_frame_indices[i]:06d}.lidar_raw.npz']['lidar_to_world']
                    lidar_points_world = (lidar_to_world @ np.concatenate([lidar_points, np.ones([lidar_points.shape[0], 1])], axis=1).T).T[:, :3]
                    accumulated_points.append(torch.tensor(lidar_points_world, device='cuda'))
                    original_bbox_dicts.append(all_object_info[f'{low_fps_frame_indices[i]:06d}.all_object_info.json'])
                    moved_bbox_dicts.append(all_object_info[f'{frame_idx:06d}.all_object_info.json'])

                accumulated_points_corrected, _ = batch_move_points_within_bboxes_sparse(
                    accumulated_points,
                    original_bbox_dicts,
                    moved_bbox_dicts,
                    torch.tensor(low_fps_frame_indices),
                    dynamic_only=True,
                    bbox_scale=1.2, # enlarge bbox to include all points
                )
                lengths_tensor = torch.tensor([t.shape[0] for t in accumulated_points_corrected], device='cuda')
                cat_points = torch.cat(accumulated_points_corrected, dim=0)
                w2cs = torch.inverse(torch.tensor(pose_data[f'{frame_idx:06d}.pose.{camera_name}.npy'])).unsqueeze(0).cuda()
                intrinsics = torch.tensor(intrinsic_this_cam).unsqueeze(0).cuda()
                depth, mask = forward_warp_multiframes_sparse_depth_only(
                    w2cs,
                    intrinsics,
                    buffer_points=cat_points.cuda(),
                    buffer_length=lengths_tensor.cuda(),
                    target_h=target_h, 
                    target_w=target_w,
                    center_only=False,
                    is_ftheta=(camera_type == 'ftheta'),
                    expanded_kernel=4
                )            
                depth = depth.squeeze(0).squeeze(0).cpu().numpy()
                lidar_depth_list.append(depth)
            lidar_depth = np.stack(lidar_depth_list, axis=0) # T, H, W
            # clamp to 75
            lidar_depth_vis = np.clip(lidar_depth, 0, 75) / 75.0 * 255
            lidar_depth_vis = lidar_depth_vis.astype(np.uint8)
            lidar_depth_vis = np.stack([lidar_depth_vis] * 3, axis=-1) # T, H, W, 3

        # center crop to crop_resolution
        if crop_resolution != target_resolution:
            crop_w, crop_h = crop_resolution
            crop_x = (target_w - crop_w) // 2
            crop_y = (target_h - crop_h) // 2
            if 'hdmap' not in skip:
                minimaps_projection_merged = minimaps_projection_merged[:, crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]
            if 'lidar' not in skip:
                lidar_depth_vis = lidar_depth_vis[:, crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]

        # save the rgb video
        if post_training:
            # load rgb
            rgb_file = os.path.join(input_root, f'{camera_type}_{camera_name}', f"{clip_id}.mp4")
            # load all frames
            vr = decord.VideoReader(rgb_file)
            num_frames = len(vr)
            all_frames = [vr[i] for i in range(num_frames)]
            # resize all frames to target_resolution
            all_frames_resized = []
            for frame_read in all_frames:
                try:
                    frame = frame_read.asnumpy()
                except AttributeError:
                    frame = frame_read.numpy()
                frame_resized = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
                # center crop to crop_resolution
                if crop_resolution != target_resolution:
                    crop_w, crop_h = crop_resolution
                    crop_x = (target_w - crop_w) // 2
                    crop_y = (target_h - crop_h) // 2
                    frame_resized = frame_resized[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]
                all_frames_resized.append(frame_resized)
        if transfer_format:
            (output_root / "videos" / camera_name).mkdir(parents=True, exist_ok=True)
            (output_root / "hdmap" / camera_name).mkdir(parents=True, exist_ok=True)
            (output_root / "lidar" / camera_name).mkdir(parents=True, exist_ok=True)
        else:
            (output_root / camera_name / output_folder / clip_id).mkdir(parents=True, exist_ok=True)
        for cur_idx, i in enumerate(range(3, len(valid_frame_ids), CUT_LEN - OVERLAP)):
            if i + CUT_LEN > len(valid_frame_ids):
                continue
            if post_training:
                # save rgb video
                rgb_writer = imageio_v1.get_writer(
                    output_root / "videos" / camera_name / f"{clip_id}_{cur_idx}.mp4" if transfer_format else \
                    output_root / camera_name / output_folder / clip_id /f"{cur_idx}_rgb.mp4",
                    fps=TARGET_RENDER_FPS,
                    codec="libx264",
                    macro_block_size=None,  # This makes sure num_frames is correct (by default it is rounded to 16x).
                    ffmpeg_params=[
                        "-crf", "18",     # Lower CRF for higher quality (0-51, lower is better)
                        "-preset", "slow",   # Slower preset for better compression/quality
                        "-pix_fmt", "yuv420p", # Ensures wide compatibility
                    ],
                )
                rgb_cut = all_frames_resized[i:i+CUT_LEN]
                for frame_idx in range(len(rgb_cut)):
                    rgb_writer.append_data(rgb_cut[frame_idx])
                rgb_writer.close()
            
            if 'hdmap' not in skip:
                minimaps_projection_merged_cut = minimaps_projection_merged[i:i+CUT_LEN]

                # save HD map condition video
                map_writer = imageio_v1.get_writer(
                    output_root / "hdmap" / camera_name / f"{clip_id}_{cur_idx}.mp4" if transfer_format else \
                        output_root / camera_name / output_folder / clip_id /f"{cur_idx}_hdmap_cond.mp4",
                    fps=TARGET_RENDER_FPS,
                    codec="libx264",
                    macro_block_size=None,  # This makes sure num_frames is correct (by default it is rounded to 16x).
                    ffmpeg_params=[
                        "-crf", "18",     # Lower CRF for higher quality (0-51, lower is better)
                        "-preset", "slow",   # Slower preset for better compression/quality
                        "-pix_fmt", "yuv420p", # Ensures wide compatibility
                    ],
                )
                for frame_idx in range(len(minimaps_projection_merged_cut)):
                    map_writer.append_data(minimaps_projection_merged_cut[frame_idx])
                map_writer.close()

            if 'lidar' not in skip:
                lidar_depth_vis_cut = lidar_depth_vis[i:i+CUT_LEN]

                # save lidar condition video
                lidar_writer = imageio_v1.get_writer(
                    output_root / "lidar" / camera_name / f"{clip_id}_{cur_idx}.mp4" if transfer_format else \
                        output_root / camera_name / output_folder / clip_id /f"{cur_idx}_lidar_cond.mp4",
                    fps=TARGET_RENDER_FPS,
                    codec="libx264",
                    macro_block_size=None,  # This makes sure num_frames is correct (by default it is rounded to 16x).
                    ffmpeg_params=[
                        "-crf", "18",     # Lower CRF for higher quality (0-51, lower is better)
                        "-preset", "slow",   # Slower preset for better compression/quality
                        "-pix_fmt", "yuv420p", # Ensures wide compatibility
                    ],
                )
                for frame_idx in range(len(lidar_depth_vis_cut)):
                    lidar_writer.append_data(lidar_depth_vis_cut[frame_idx])
                lidar_writer.close()

            
@click.command()
@click.option("--input_root", '-i', type=str, help="the root folder of the input data")
@click.option("--output_root", '-o', type=str, help="the root folder of the output data")
@click.option("--dataset", "-d", type=str, default="rds_hq", help="the dataset name, 'rds_hq' or 'waymo' or 'waymo_mv', see the config in settings.json")
@click.option("--camera_type", "-c", type=str, default="ftheta", help="the type of camera model, 'pinhole' or 'ftheta'")
@click.option("--skip", "-s", multiple=True, help="can be 'hdmap' or 'lidar'")
@click.option("--output_folder", "-f", type=str, default="render", help="Output folder")
@click.option("--post_training", "-p", type=bool, default=False, help="if True, output the RGB video for post-training")
@click.option("--transfer_format", type=bool, default=False, help="output in transfer1 folder structure")
@click.option("--num", "-n", type=int, default=-1, help="num clips to process")
def main(input_root, output_root, dataset, camera_type, skip, output_folder, post_training, transfer_format, num):
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    setup(local_rank, world_size)

    if skip is not None:
        # assert skip is a list of 'hdmap' or 'bbox'
        assert all(s in ['hdmap', 'lidar'] for s in skip), "skip must be in ['hdmap', 'lidar']"

    if post_training: # for post-training only
        assert dataset in ['waymo', 'waymo_mv'], "post_training is only supported for waymo dataset"
        assert camera_type == 'pinhole', "post_training is only supported for pinhole camera"

    # Load settings
    with open(f'config/dataset_{dataset}.json', 'r') as file:
        settings = json.load(file)

    # get all clip ids
    input_root_p = Path(input_root)
    output_root_p = Path(output_root)
    clip_list = (input_root_p / 'pose').rglob('*.tar')
    clip_list = [c.stem for c in clip_list]
    if num > 0:
        clip_list = clip_list[:num]

    # shuffle the clip list
    np.random.seed(0)
    np.random.shuffle(clip_list)

    # distribute the clip list to each process
    clip_list = clip_list[local_rank::world_size]

    for clip_id in tqdm(clip_list):
        render_sample(
            input_root = input_root,
            output_root = output_root_p,
            clip_id = clip_id,
            settings = settings,
            camera_type = camera_type,
            skip = skip,
            output_folder = output_folder,
            post_training = post_training,
            transfer_format=transfer_format
        )

if __name__ == "__main__":
    main()