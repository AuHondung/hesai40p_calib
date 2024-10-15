#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <Eigen/Dense>
#include "rpzCalib.h"


class Lidar2CarNode {
public:
    Lidar2CarNode(ros::NodeHandle& nh) {
        ground_plane_pub_ = nh.advertise<sensor_msgs::PointCloud2>("ground_plane_cloud", 10);
        calib_sub_ = nh.subscribe("/hesai/pandar", 10, &Lidar2CarNode::CalibCallback, this);
    }

    void CalibCallback(const sensor_msgs::PointCloud2ConstPtr& cloud_msg) {
        PointCloudPtr cloud(new PointCloud());
        pcl::fromROSMsg(*cloud_msg, *cloud);

        PointCloudPtr master_gcloud = rp_calib_.Calibrate(cloud);

        if (!master_gcloud->empty()) {
            sensor_msgs::PointCloud2 output_msg;
            pcl::toROSMsg(*master_gcloud, output_msg);
            output_msg.header = cloud_msg->header; // 保持相同的帧 ID 和时间戳
            ground_plane_pub_.publish(output_msg);
        }
    }
private:
    ros::Subscriber calib_sub_;
    ros::Publisher ground_plane_pub_;
    RPCalib rp_calib_;
};

int main(int argc, char **argv) {
    // 初始化ROS节点
    ros::init(argc, argv, "lidar2car_node");
    ros::NodeHandle nh;

    // 创建节点对象
    Lidar2CarNode lidar2car_node(nh);

    ros::spin();
    return 0;
}

// int main(int argc, char **argv) {
//     // 初始化ROS节点
//     ros::init(argc, argv, "lidar2car_node");
//     ros::NodeHandle nh;
//     ros::Subscriber calib_sub = nh.subscribe("/hesai/pandar", 10, CalibCallback);

//     // RPCalib rp_calib("./output");

//     // // 加载pcd数据
//     // std::string dataset_folder = "./data/2024-08-23-15-50-10_ground_pcd/";
//     // std::vector<Eigen::Matrix4d> lidar_pose;  
//     // int start_frame = 0;
//     // int end_frame = 100;
//     // if (!rp_calib.LoadData(dataset_folder, lidar_pose, start_frame, end_frame)) {
//     //     ROS_ERROR("Failed to load data.");
//     //     return -1;
//     // }

//     // Eigen::Matrix4d extrinsic;
//     // if (!rp_calib.Calibrate(extrinsic)) {
//     //     ROS_ERROR("Calibration failed.");
//     //     return -1;
//     // }

//     // ROS_INFO("Calibration completed successfully.");

//     ros::spin();
//     return 0;
// }
