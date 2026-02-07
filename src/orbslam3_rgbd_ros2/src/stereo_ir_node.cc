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

class ORBSLAM3StereoIRNode : public rclcpp::Node
{
public:
  ORBSLAM3StereoIRNode()
  : Node("orbslam3_stereo_ir_node")
  {
    // ---------------- Params ----------------
    vocab_path_    = this->declare_parameter<std::string>("vocab_path", "");
    settings_path_ = this->declare_parameter<std::string>("settings_path", "");

    left_topic_    = this->declare_parameter<std::string>("left_topic",
                      "/camera/camera/infra1/image_rect_raw");
    right_topic_   = this->declare_parameter<std::string>("right_topic",
                      "/camera/camera/infra2/image_rect_raw");

    out_dir_       = this->declare_parameter<std::string>("out_dir", "./ply_out");
    final_ply_name_= this->declare_parameter<std::string>("final_ply_name", "final.ply");

    publish_tf_    = this->declare_parameter<bool>("publish_tf", true);
    tf_child_frame_= this->declare_parameter<std::string>("tf_child_frame", "camera");
    map_frame_     = this->declare_parameter<std::string>("map_frame", "map");

    path_stride_   = this->declare_parameter<int>("path_stride", 1);
    // ----------------------------------------

    if (vocab_path_.empty() || settings_path_.empty()) {
      RCLCPP_ERROR(get_logger(), "vocab_path or settings_path is empty. Please set ROS params.");
      throw std::runtime_error("Missing ORB-SLAM3 params");
    }

    // output directory
    try {
      std::filesystem::create_directories(out_dir_);
    } catch (const std::exception& e) {
      RCLCPP_WARN(get_logger(), "Failed to create out_dir (%s): %s", out_dir_.c_str(), e.what());
    }

    // ---------------- ORB-SLAM3 ----------------
    // Stereo mode (no depth image)
    // viewer 你现在不需要的话建议 false（但你已经走了 pangolin 链接路线，true/false都能编）
    slam_ = std::make_unique<ORB_SLAM3::System>(
      vocab_path_, settings_path_, ORB_SLAM3::System::STEREO, true);

    // ---------------- Publishers ----------------
    pose_pub_ = this->create_publisher<geometry_msgs::msg::PoseStamped>("/orbslam3/camera_pose", 10);
    path_pub_ = this->create_publisher<nav_msgs::msg::Path>("/orbslam3/path", 10);

    path_msg_.header.frame_id = map_frame_;

    if (publish_tf_) {
      tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);
    }

    // ---------------- Subscribers (sync L/R) ----------------
    left_sub_  = std::make_shared<message_filters::Subscriber<sensor_msgs::msg::Image>>(this, left_topic_);
    right_sub_ = std::make_shared<message_filters::Subscriber<sensor_msgs::msg::Image>>(this, right_topic_);

    using SyncPolicy = message_filters::sync_policies::ApproximateTime<
      sensor_msgs::msg::Image, sensor_msgs::msg::Image>;

    sync_ = std::make_shared<message_filters::Synchronizer<SyncPolicy>>(
      SyncPolicy(30), *left_sub_, *right_sub_);

    sync_->registerCallback(std::bind(&ORBSLAM3StereoIRNode::stereoCallback, this, _1, _2));

    RCLCPP_INFO(get_logger(),
                "ORB-SLAM3 Stereo-IR node started.\n  Left:  %s\n  Right: %s\n  Out:   %s/%s\n  TF:    %s",
                left_topic_.c_str(),
                right_topic_.c_str(),
                out_dir_.c_str(),
                final_ply_name_.c_str(),
                publish_tf_ ? "ON" : "OFF");
  }

  ~ORBSLAM3StereoIRNode() override
  {
    if (slam_) {
      RCLCPP_INFO(get_logger(), "Shutting down ORB-SLAM3...");
      slam_->Shutdown();

      const std::string final_path = out_dir_ + "/" + final_ply_name_;
      slam_->SavePointCloud(final_path);

      RCLCPP_INFO(get_logger(), "Saved FINAL PLY: %s", final_path.c_str());
    }
  }

private:
  void stereoCallback(const sensor_msgs::msg::Image::ConstSharedPtr& left_msg,
                      const sensor_msgs::msg::Image::ConstSharedPtr& right_msg)
  {
    const double t = rclcpp::Time(left_msg->header.stamp).seconds();

    // IR 通常是单通道，优先用 mono8
    cv::Mat imL, imR;
    try {
      // 如果本来就是 mono8，这样最稳
      imL = cv_bridge::toCvCopy(left_msg, "mono8")->image;
      imR = cv_bridge::toCvCopy(right_msg, "mono8")->image;
    } catch (const std::exception &e) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "cv_bridge IR error: %s", e.what());
      return;
    }

    // ORB-SLAM3 Stereo tracking
    Sophus::SE3f Tcw = slam_->TrackStereo(imL, imR, t);
    Sophus::SE3f Twc = Tcw.inverse();

    const Eigen::Vector3f twc = Twc.translation();
    const Eigen::Matrix3f Rwc = Twc.rotationMatrix();
    Eigen::Quaternionf qwc(Rwc);

    // PoseStamped
    geometry_msgs::msg::PoseStamped ps;
    ps.header.stamp = left_msg->header.stamp;
    ps.header.frame_id = map_frame_;
    ps.pose.position.x = twc.x();
    ps.pose.position.y = twc.y();
    ps.pose.position.z = twc.z();
    ps.pose.orientation.x = qwc.x();
    ps.pose.orientation.y = qwc.y();
    ps.pose.orientation.z = qwc.z();
    ps.pose.orientation.w = qwc.w();

    pose_pub_->publish(ps);

    // Path
    frame_count_++;
    if (path_stride_ <= 1 || (frame_count_ % path_stride_ == 0)) {
      path_msg_.header.stamp = left_msg->header.stamp;
      path_msg_.poses.push_back(ps);
      path_pub_->publish(path_msg_);
    }

    // TF (map -> camera)
    if (publish_tf_ && tf_broadcaster_) {
      geometry_msgs::msg::TransformStamped tf;
      tf.header.stamp = left_msg->header.stamp;
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
  std::unique_ptr<ORB_SLAM3::System> slam_;

  std::shared_ptr<message_filters::Subscriber<sensor_msgs::msg::Image>> left_sub_;
  std::shared_ptr<message_filters::Subscriber<sensor_msgs::msg::Image>> right_sub_;
  std::shared_ptr<message_filters::Synchronizer<
    message_filters::sync_policies::ApproximateTime<
      sensor_msgs::msg::Image, sensor_msgs::msg::Image>>> sync_;

  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pose_pub_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_pub_;
  nav_msgs::msg::Path path_msg_;
  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;

  std::string vocab_path_, settings_path_;
  std::string left_topic_, right_topic_;
  std::string out_dir_, final_ply_name_;
  bool publish_tf_{true};
  std::string tf_child_frame_, map_frame_;
  int path_stride_{1};
  int frame_count_{0};
};

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<ORBSLAM3StereoIRNode>());
  rclcpp::shutdown();
  return 0;
}
