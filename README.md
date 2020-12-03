# Livox Lane Detection

![avatar](./result/demo/demo.gif)
# Introduction
This repository serves as a inference suite for [Livox](https://www.livoxtech.com/cn/) point cloud lane detection. It supports semantic segmentation of general lane line types and objects near the road.

# Dependencies
- `Python3.6+`
- `Pytorch1.0+` (tested on 1.4.0)
- `OpenCV Python`
- `Numpy`

# Citing
If you find this repository useful, please consider citing it using a link to the repo :)

# Files and Direcories
- **test_lane_detection.py:**  Testing lane detection
- **visualize_points_with_class.py:**  Visualizing the points with semantic specific colors.
- **config.py:**  Parameter configurations used in this repository.
- **data_process:**  Folder containing data processing scripts for point cloud.
- **model:**  Folder containing model files.
- **network:**  Folder containing network architecure implementations.
 

# Usage
### 1. Quick Start
We use the data format as in [Livox Dataset V1.0](https://www.livoxtech.com/cn/dataset). There's an example in `test_data` folder. To test, run directly:
```bash
$ python test_lane_detection.py
```
The lane detection results are saved in `result/points_with_class`.

You can visualize the results by:
```bash
$ python visualize_points_with_class.py
```
The visualized results are saved in `result/points_vis`.

### 2. Configure for Your Need
The configuration parameters used in this repository are listed in `config.py`. The parameter details are as follows:
```
LIDAR_IDs                          # Selected lidar ids
BV_COMMON_SETTINGS = {             # Bird view map setttings
"train_height_shift",              # Height shift to make the z-axis value of ground be 0
"shifted_min_height",              # Minimum z-axis value of the interval to select points near the ground
"shifted_max_height",              # Maximum z-axis value of the interval to select points near the ground
"distance_resolution_train",       # 1 meter in x-axis corresponds to "distance_resolution_train" pixels on the bird view map
"width_resolution_train",          # 1 meter in y-axis corresponds to "width_resolution_train" pixels on the bird view map
"point_radius_train",              # point radius on the bird view
"truncation_max_intensiy",         # If intensity value of one point is bigger than "truncation_max_intensiy", the intensity will be set to this value
"train_background_intensity_shift",# Intensity shift to make the area with points (intensity may be 0) different with that without points
}
BV_RANGE_SETTINGS = { 
"max_distance",                    # Farthest detection distnace in front of the car
"min_distance",                    # Farthest detection distnace behind the car
"left_distance",                   # Farthest detection distance to the left of the car
"right_distance"                   # Farthest detection distance to the right of the car
}
MODEL_NAME                         # Model name
GPU_IDs                            # GPU
TEST_DATA_FOLDER                   # Path to input data
POINTS_WITH_CLASS_FOLDER           # Path to lane detection results
VIS_FOLDER                         # Path to visulization results
```

#### 2.1 Test on Livox Dataset V1.0
Download the [Livox Dataset V1.0](https://www.livoxtech.com/cn/dataset). Unzip it to some folder. Then specify the "TEST_DATA_FOLDER" in `config.py` to that folder. You can also specify the "POINTS_WITH_CLASS_FOLDER" and "VIS_FOLDER" for your convenience.

#### 2.2 Change the Lidar Configuration
The Livox Dataset V1.0 perceive the environment using 6 lidars. The default setting is to use all of them for lane detection. If you want to do lane detection using a few of the lidars, you can change the "LIDAR_IDs" in `config.py` for your need.
For example, if you want to detect the lane in front of the car, you can configure as: 
```bash
LIDAR_IDs = ["1", "6"]
```
