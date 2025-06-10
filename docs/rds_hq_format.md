# RDS-HQ Format

If the file extension is ‘.tar’, you can use the following get_sample function to extract its data in Python

```bash
!pip install webdataset

from webdataset import WebDataset, non_empty

def get_sample(file_path):
    dataset = WebDataset(file_path, nodesplitter=non_empty, workersplitter=None, shardshuffle=False).decode()
    return next(iter(dataset))

data = get_sample(...)
```


## Sensor Data

| Folder | File Format | Description | Explanation | 
| ----- | ----- | ----- | ----- |
| lidar\_raw | .tar | Motion-compensated LiDAR point clouds (10 FPS) | keys include `000000.lidar_raw.npz`, `000003.lidar_raw.npz`, ..., etc. The `000000.lidar_raw.npz` includes `xyz`, `intensity`, `row`, `column`, `starting_timestamp`, `lidar_to_world` | 
| vehicle\_pose | .tar | Vehicle poses (30 FPS) in FLU convention | keys include `000000.vehicle_pose.npy`, `000001.vehicle_pose.npy`, etc. |
| pose | .tar | Camera poses derived from vehicle pose (30 FPS) in OpenCV convention | keys include `000000.pose.{camera_name}.npy`, `000001.pose.{camera_name}.npy`, etc. |
| ftheta\_intrinsic | .tar | Camera intrinsic parameters for each view | keys include `ftheta_intrinsic.{camera_name}.npy`. The npy file stores a vector `[cx, cy, w, h, *poly, is_bw_poly, *linear_cde]`. `*poly` includes 6 polynomial parameters for f-theta camera, `is_bw_poly` indicates if it is a backward polynomial, `*linear_cde` includes 3 parameters for f-theta camera, by default it is `[1, 0, 0]` and can be omitted. |
| pinhole\_intrinsic | .tar | Pinhole camera intrinsic parameters for each view (for rectification) | keys include `pinhole_intrinsic.{camera_name}.npy`. The npy file stores a vector `[fx, fy, cx, cy, w, h]`.  |
| car_mask_coarse | .png | A coarse mask for the vehicle hood | pixel value > 0 means hood area.|

`{camera_name}` includes 
- camera_front_wide_120fov
- camera_cross_left_120fov
- camera_cross_right_120fov
- camera_rear_left_70fov
- camera_rear_right_70fov
- camera_rear_tele_30fov
- camera_front_tele_30fov


## HDMap Annotations

| Folder | File Format | Description | Explanation | 
| ----- | ----- | ----- | ----- |
| 3d\_lanes | .tar | 3D lane boundaries (left and right), polyline format | keys include `lanes.json`. You can access the left and right boundaries via `['lanes.josn']['labels'][0/1/2/…]['labelData']['shape3d']['polylines3d']['polylines']`. Here `['lanes.json']['labels']` is a list, includes many left-right lane pairs. |
| 3d\_lanelines | .tar | 3D lane centerlines, polyline format | keys include `lanelines.json`. Laneline is the center of left and right lanes. You can access the vertices via `['lanelines.json']['labels'][0/1/2/…]['labelData']['shape3d']['polyline3d']['vertices']` | 
| 3d\_road\_boundaries | .tar | Road boundary annotations, polyline format | keys include `road_boundaries.json`. You can access the vertices via `['road_boundaries.json']['labels'][0/1/2/…]['labelData']['shape3d']['polyline3d']['vertices']` | 
| 3d\_wait\_lines | .tar | Waiting lines at intersections, polyline format | keys include `wait_lines.json`. You can access the vertices via `['wait_lines.json']['labels'][0/1/2/…]['labelData']['shape3d']['polyline3d']['vertices']` | 
| 3d\_crosswalks | .tar | Crosswalk annotations, polygon format | keys include `crosswalks.json`. You can access the vertices via `['crosswalks.json']['labels'][0/1/2/…]['labelData']['shape3d']['surface']['vertices']` |
| 3d\_road\_markings | .tar | Road surface markings (turning arrows, stop lines, etc.), polygon format | keys include `road_markings.json`. You can access the vertices via `['road_markings.json']['labels'][0/1/2/…]['labelData']['shape3d']['surface']['vertices']` |
| 3d\_poles | .tar | Traffic poles, polyline format | keys include `poles.json`. You can access the vertices via `['poles.json']['labels'][0/1/2/…]['labelData']['shape3d']['polyline3d']['vertices']` | 
| 3d\_traffic\_lights | .tar | Traffic lights, 3D cuboid format | keys include `3d_traffic_lights.json`. You can access 8 corner vertices via `['3d_traffic_lights.json']['labels'][0/1/2/…]['labelData']['shape3d']['cuboid3d']['vertices']` | 
| 3d\_traffic\_signs | .tar | Traffic signs, 3D cuboid format | keys include `3d_traffic_signs.json`. You can access 8 corner vertices via `['3d_traffic_signs.json']['labels'][0/1/2/…]['labelData']['shape3d']['cuboid3d']['vertices']` | 

## Dynamic Object Annotations

| Folder | File Format | Description |  Explanation | 
| ----- | ----- | ----- | ----- | 
| all\_object\_info | .tar | 4D object tracking (position, dimensions, movement state) | keys include `000000.all_object_info.json`, `000003.all_object_info.json`, etc. For `000000.all_object_info.json`, they store `{tracking_id :{'object_to_world': 4x4 transformation matrix, 'object_lwh': [length, width, height], 'object_is_moving': True or False, 'object_type': str }}` |

For 10 Examples in [Cosmos-Transfer1-7B-Sample-AV-Data-Example](https://huggingface.co/datasets/nvidia/Cosmos-Transfer1-7B-Sample-AV-Data-Example), object type includes
```
['Unknown', 'Ground', 'GroundPaint', 'LaneLine', 'RoadDots', 'StopLine', 'CrossWalk', 'RoadSymbol', 'ParkingDelimeter', 'Zone', 'SpeedBump', 'Surface', 'Asphalt', 'Dirt', 'Gravel', 'Vehicle', 'Car', 'Bus', 'Truck', 'Train', 'Ego', 'MovableObject', 'Pedestrian', 'Cyclist', 'EgoVehicle', 'Vegetation', 'TreeTrunk', 'Wall', 'Building', 'Bridge', 'Tunnel', 'RoadworkObject', 'TrafficCone', 'TrafficSign', 'TrafficSignal', 'Pole', 'VerticalPole', 'Walkable', 'Sidewalk', 'Stairs', 'Boundary', 'TrafficDivider', 'Curb', 'Barrier', 'GuardRail', 'Other', 'Wire', 'Reflection', 'Sky']
```

For labels in [PhysicalAI-Autonomous-Vehicle-Cosmos-Synthetic](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicle-Cosmos-Synthetic) , object type includes
- Automobile
- Heavy_truck
- Bus
- Train_or_tram_car
- Trolley_bus
- Other_vehicle
- Trailer
- Person
- Stroller
- Rider
- Animal
- Protruding_object


## Camera and LiDAR Synchronization
- Camera Frame Rate: 30 FPS
- LiDAR Frame Rate: 10 FPS
- Synchronization: Each LiDAR frame corresponds to 3 consecutive camera frames.
- Pose Interpolation: Camera poses are interpolated at the starting timestamp of each image frame.
