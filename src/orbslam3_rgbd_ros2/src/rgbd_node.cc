#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <cv_bridge/cv_bridge.h>

#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <nav_msgs/msg/path.hpp>
#include <tf2_ros/transform_broadcaster.h>

#include <opencv2/core/core.hpp>

#include <filesystem>
#include <memory>
#include <string>

#include "System.h"   // ORB-SLAM3 System

using std::placeholders::_1;
using std::placeholders::_2;

class ORBSLAM3RGBDNode : public rclcpp::Node
{
public:
  ORBSLAM3RGBDNode()
  : Node("orbslam3_rgbd_node")
  {
    // ---------------- Params ----------------
    vocab_path_    = this->declare_parameter<std::string>("vocab_path", "");
    settings_path_ = this->declare_parameter<std::string>("settings_path", "");
    rgb_topic_     = this->declare_parameter<std::string>("rgb_topic", "/camera/camera/color/image_raw");
    depth_topic_   = this->declare_parameter<std::string>("depth_topic", "/camera/camera/aligned_depth_to_color/image_raw");
    out_dir_       = this->declare_parameter<std::string>("out_dir", "./ply_out");
    final_ply_name_= this->declare_parameter<std::string>("final_ply_name", "final.ply");

    publish_tf_    = this->declare_parameter<bool>("publish_tf", true);
    tf_child_frame_= this->declare_parameter<std::string>("tf_child_frame", "camera");
    map_frame_     = this->declare_parameter<std::string>("map_frame", "map");

    // Path 发布频率：每 N 帧追加一个 pose 到 path（避免 path 太密卡 RViz）
    path_stride_   = this->declare_parameter<int>("path_stride", 1);

    // 重要：建议你播放 bag 时用 --clock，这里就要 use_sim_time=true
    // 你可以在命令行传 -p use_sim_time:=true
    // ----------------------------------------

    if (vocab_path_.empty() || settings_path_.empty()) {
      RCLCPP_ERROR(get_logger(), "vocab_path or settings_path is empty. Please set ROS params.");
      throw std::runtime_error("Missing ORB-SLAM3 params");
    }

    // Create output directory
    try {
      std::filesystem::create_directories(out_dir_);
    } catch (const std::exception& e) {
      RCLCPP_WARN(get_logger(), "Failed to create out_dir (%s): %s", out_dir_.c_str(), e.what());
    }

    // ---------------- ORB-SLAM3 ----------------
    // RGBD mode
    slam_ = std::make_unique<ORB_SLAM3::System>(
      vocab_path_, settings_path_, ORB_SLAM3::System::RGBD, true);

    // ---------------- Publishers ----------------
    pose_pub_ = this->create_publisher<geometry_msgs::msg::PoseStamped>("/orbslam3/camera_pose", 10);
    path_pub_ = this->create_publisher<nav_msgs::msg::Path>("/orbslam3/path", 10);

    path_msg_.header.frame_id = map_frame_;

    if (publish_tf_) {
      tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);
    }

    // ---------------- Subscribers (sync) ----------------
    rgb_sub_   = std::make_shared<message_filters::Subscriber<sensor_msgs::msg::Image>>(this, rgb_topic_);
    depth_sub_ = std::make_shared<message_filters::Subscriber<sensor_msgs::msg::Image>>(this, depth_topic_);

    using SyncPolicy = message_filters::sync_policies::ApproximateTime<
      sensor_msgs::msg::Image, sensor_msgs::msg::Image>;

    sync_ = std::make_shared<message_filters::Synchronizer<SyncPolicy>>(
      SyncPolicy(30), *rgb_sub_, *depth_sub_);

    sync_->registerCallback(std::bind(&ORBSLAM3RGBDNode::rgbdCallback, this, _1, _2));

    RCLCPP_INFO(get_logger(),
                "ORB-SLAM3 RGBD node started.\n  RGB:   %s\n  Depth: %s\n  Out:   %s/%s\n  TF:    %s",
                rgb_topic_.c_str(),
                depth_topic_.c_str(),
                out_dir_.c_str(),
                final_ply_name_.c_str(),
                publish_tf_ ? "ON" : "OFF");
  }

  ~ORBSLAM3RGBDNode() override
  {
    // 结束时导出一次 ply
    if (slam_) {
      RCLCPP_INFO(get_logger(), "Shutting down ORB-SLAM3...");
      slam_->Shutdown();

      const std::string final_path = out_dir_ + "/" + final_ply_name_;
      slam_->SavePointCloud(final_path);

      RCLCPP_INFO(get_logger(), "Saved FINAL PLY: %s", final_path.c_str());
    }
  }

private:
  void rgbdCallback(const sensor_msgs::msg::Image::ConstSharedPtr& rgb_msg,
                    const sensor_msgs::msg::Image::ConstSharedPtr& depth_msg)
  {
    // ---------------- Timestamp ----------------
    // 用 header stamp（播放 bag --clock + use_sim_time=true 时很关键）
    const double t = rclcpp::Time(rgb_msg->header.stamp).seconds();

    // ---------------- RGB ----------------
    cv::Mat rgb;
    try {
      // 强制 bgr8（颜色通道顺序对 SLAM 本身影响不大，但更统一）
      rgb = cv_bridge::toCvCopy(rgb_msg, "bgr8")->image;
    } catch (const std::exception &e) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "cv_bridge RGB error: %s", e.what());
      return;
    }

    // ---------------- Depth ----------------
    cv::Mat depth_m;
    try {
      auto cv_ptr = cv_bridge::toCvCopy(depth_msg);
      const std::string enc = depth_msg->encoding;

      if (enc == "16UC1") {
        // 16U depth 通常是 mm，转成 float 米（最不坑）
        cv_ptr->image.convertTo(depth_m, CV_32F, 1.0 / 1000.0);
      } else if (enc == "32FC1") {
        depth_m = cv_ptr->image;
      } else {
        // 不常见编码：尽量继续跑，但给出提示
        depth_m = cv_ptr->image;
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                             "Unexpected depth encoding: %s (continue anyway)", enc.c_str());
      }
    } catch (const std::exception &e) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "cv_bridge Depth error: %s", e.what());
      return;
    }

    // ---------------- ORB-SLAM3 TrackRGBD ----------------
    // 注意：ORB-SLAM3 通常返回 Tcw（camera pose wrt world，或 world->camera），
    // 这里我们用 inverse() 得到 Twc（camera in world），更适合 RViz “相机在地图里运动”的直觉显示。
    Sophus::SE3f Tcw = slam_->TrackRGBD(rgb, depth_m, t);
    Sophus::SE3f Twc = Tcw.inverse();

    // ---------------- Publish PoseStamped ----------------
    const Eigen::Vector3f twc = Twc.translation();
    const Eigen::Matrix3f Rwc = Twc.rotationMatrix();
    Eigen::Quaternionf qwc(Rwc);

    geometry_msgs::msg::PoseStamped ps;
    ps.header.stamp = rgb_msg->header.stamp;
    ps.header.frame_id = map_frame_;
    ps.pose.position.x = twc.x();
    ps.pose.position.y = twc.y();
    ps.pose.position.z = twc.z();
    ps.pose.orientation.x = qwc.x();
    ps.pose.orientation.y = qwc.y();
    ps.pose.orientation.z = qwc.z();
    ps.pose.orientation.w = qwc.w();

    pose_pub_->publish(ps);

    // ---------------- Publish Path ----------------
    frame_count_++;
    if (path_stride_ <= 1 || (frame_count_ % path_stride_ == 0)) {
      path_msg_.header.stamp = rgb_msg->header.stamp;
      path_msg_.poses.push_back(ps);
      path_pub_->publish(path_msg_);
    }

    // ---------------- Publish TF (map -> camera) ----------------
    if (publish_tf_ && tf_broadcaster_) {
      geometry_msgs::msg::TransformStamped tf;
      tf.header.stamp = rgb_msg->header.stamp;
      tf.header.frame_id = map_frame_;
      tf.child_frame_id = tf_child_frame_;

      tf.transform.translation.x = twc.x();
      tf.transform.translation.y = twc.y();
      tf.transform.translation.z = twc.z();
      tf.transform.rotation.x = qwc.x();
      tf.transform.rotation.y = qwc.y();
      tf.transform.rotation.z = qwc.z();
      tf.transform.rotation.w = qwc.w();

      tf_broadcaster_->sendTransform(tf);
    }
  }

private:
  // ORB-SLAM3
  std::unique_ptr<ORB_SLAM3::System> slam_;

  // message_filters
  std::shared_ptr<message_filters::Subscriber<sensor_msgs::msg::Image>> rgb_sub_;
  std::shared_ptr<message_filters::Subscriber<sensor_msgs::msg::Image>> depth_sub_;
  std::shared_ptr<message_filters::Synchronizer<
    message_filters::sync_policies::ApproximateTime<
      sensor_msgs::msg::Image, sensor_msgs::msg::Image>>> sync_;

  // publishers
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pose_pub_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_pub_;
  nav_msgs::msg::Path path_msg_;

  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;

  // params
  std::string vocab_path_, settings_path_;
  std::string rgb_topic_, depth_topic_;
  std::string out_dir_, final_ply_name_;
  bool publish_tf_{true};
  std::string tf_child_frame_, map_frame_;
  int path_stride_{1};

  int frame_count_{0};
};

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<ORBSLAM3RGBDNode>());
  rclcpp::shutdown();
  return 0;
}
