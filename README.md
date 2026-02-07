使用方法

```bash
ros2 bag play ~/包路径只到文件夹即可 --clock
```

```bash
ros2 run orbslam3_rgbd_ros2 rgbd_node --ros-args \
  -p use_sim_time:=true \
  -p vocab_path:=/到ORB_SLAM3文件夹的路径/Vocabulary/ORBvoc.txt \
  -p settings_path:=/到ORB_SLAM3文件夹的路径/Examples/RGB-D/RealSense_D435i.yaml \
  -p rgb_topic:=/camera/camera/color/image_raw \
  -p depth_topic:=/camera/camera/aligned_depth_to_color/image_raw 
```
