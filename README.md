# DCL-SLAM

A ROS package of DCL-SLAM: Distributed Collaborative LiDAR SLAM Framework for a Robotic Swarm. 

https://user-images.githubusercontent.com/41199568/213071890-679025cf-23f5-48f0-a2ef-d9b00911f926.mp4

The HD video of the demonstration of DCL-SLAM is avaliable at [BiliBili](https://www.bilibili.com/video/BV12G4y187mw/?spm_id_from=333.337.search-card.all.click).

## Prerequisites
  - [Ubuntu ROS](http://wiki.ros.org/ROS/Installation) (Robot Operating System on Ubuntu 18.04 or 20.04)
  - Python (For wstool and catkin tool)
  - CMake (Compilation Configuration Tool)
  - [Boost](http://www.boost.org/) (portable C++ source libraries)
  - [PCL](https://pointclouds.org/downloads/linux.html) (Default Point Cloud Library on Ubuntu work normally)
  - [Eigen](http://eigen.tuxfamily.org/index.php?title=Main_Page) (Default Eigen library on Ubuntu work normally)
  ```
  sudo apt-get install cmake libboost-all-dev python-wstool python-catkin-tools
  ```
  - [GTSAM](https://github.com/borglab/gtsam/releases) (Georgia Tech Smoothing and Mapping library)
  - [livox_ros_driver](https://github.com/Livox-SDK/livox_ros_driver) (The driver for Livox LiDAR)

  These prerequisites will be installed during the compilation.

## Compilation
  Set up the workspace configuration:
  ```
  mkdir -p ~/cslam_ws/src
  cd ~/cslam_ws
  catkin init
  catkin config --merge-devel
  catkin config --cmake-args -DCMAKE_BUILD_TYPE=Release
  ```
  
  Then use wstool for fetching catkin dependencies:
  ```
  cd src
  git clone https://github.com/PengYu-Team/DCL-SLAM.git
  git clone https://github.com/PengYu-Team/DCL-LIO-SAM.git
  git clone https://github.com/PengYu-Team/DCL-FAST-LIO.git
  wstool init
  wstool merge DCL-SLAM/dependencies.rosinstall
  wstool update
  ```

  Build DCL-SLAM
  ```
  catkin build dcl_lio_sam
  catkin build dcl_fast_lio
  ```
  
## Run with Dataset
  - [S3E dataset](https://github.com/PengYu-Team/S3E). The datasets are configured to run with default parameter.
  ```
  roslaunch dcl_slam run.launch
  rosbag play *your-bag-path*.bag
  ```
  
  - Other dataset. Please follow [LIO-SAM](https://github.com/TixiaoShan/LIO-SAM) and [FAST-LIO2](https://github.com/hku-mars/FAST_LIO) to set your own config file for the dataset in "config/*your-config-file*.yaml", and change the path in "launch/single_ugv.launch".

## Citation
This work is published in IEEE Sensors Journal 2023, and please cite related papers:

```
@ARTICLE{10375928,
  author={Zhong, Shipeng and Qi, Yuhua and Chen, Zhiqiang and Wu, Jin and Chen, Hongbo and Liu, Ming},
  journal={IEEE Sensors Journal}, 
  title={DCL-SLAM: A Distributed Collaborative LiDAR SLAM Framework for a Robotic Swarm}, 
  year={2024},
  volume={24},
  number={4},
  pages={4786-4797},
  keywords={Simultaneous localization and mapping;Laser radar;Sensors;Optimization;Feature extraction;Trajectory;Odometry;Collaborative localization;distributed framework;place recognition;range sensor},
  doi={10.1109/JSEN.2023.3345541}}
```

```
@article{feng2022s3e,
  title={S3e: A large-scale multimodal dataset for collaborative slam},
  author={Feng, Dapeng and Qi, Yuhua and Zhong, Shipeng and Chen, Zhiqiang and Jiao, Yudu and Chen, Qiming and Jiang, Tao and Chen, Hongbo},
  journal={arXiv preprint arXiv:2210.13723},
  year={2022}
}
```

## Acknowledgement

  - DCL-LIO-SAM adopt [LIO-SAM](https://github.com/TixiaoShan/LIO-SAM) as front end (Shan, Tixiao and Englot, Brendan and Meyers, Drew and Wang, Wei and Ratti, Carlo and Rus Daniela. LIO-SAM: Tightly-coupled Lidar Inertial Odometry via Smoothing and Mapping).

  - DCL-FAST-LIO adopt [FAST-LIO2](https://github.com/hku-mars/FAST_LIO) as front end (Xu, Wei, Yixi Cai, Dongjiao He, Jiarong Lin, and Fu Zhang. Fast-lio2: Fast direct lidar-inertial odometry).

  - DCL-SLAM is based on a two-stage distributed Gauss-Seidel approach (Siddharth Choudhary and Luca Carlone and Carlos Nieto and John Rogers and Henrik I. Christensen and Frank Dellaert. Distributed Trajectory Estimation with Privacy and Communication Constraints: a Two-Stage Distributed Gauss-Seidel Approach).

  - DCL-SLAM is based on outlier rejection of [DOOR-SLAM](https://github.com/lajoiepy/robust_distributed_mapper) (Lajoie, Pierre-Yves and Ramtoula, Benjamin and Chang, Yun and Carlone, Luca and Beltrame, Giovanni. DOOR-SLAM: Distributed, Online, and Outlier Resilient SLAM for Robotic Teams).


# Acknowledgement

Part of the source code in this repository is developed within the frame and for the purpose of the OpenSwarm project. This project has received funding from the European Unioan's Horizon Europe Framework Programme under Grant Agreement No. 101093046.

![OpenSwarm - Funded by the European Union](logos/ack.png)
