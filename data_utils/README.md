# StereoOccupancy

## Requirements

## Data
运行data_utils/depth-anything-tensorrt-main/run_batch.bash，生成六视图的深度值，环境dpth_cpp

运行data_utils/generate_occupancy_nuscenes/run_gt_cpu.sh，生成六视图Occupancy值，环境/mnt/data/home/fuquan.jin/miniconda/envs/triro_optix

运行data_utils/gen_depth_gt.py，将LiDAR点云映射到相机图像中，生成深度图

stereo-from-mono/drivingstereo_dataset.py，根据深度图和右目图生成左目图
## Demo

## Evaluation

## Train