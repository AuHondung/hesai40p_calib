#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <tf2_eigen/tf2_eigen.h>
#include <geometry_msgs/TransformStamped.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/common/transforms.h>
#include <pcl/visualization/cloud_viewer.h>

#include "GroundExtract.h"
#include "PlateExtract.h"

GroundExtractor ground_extractor;
PlateExtractor plate_extractor;

int frame_count = 0;
const int max_frames = 1;
pcl::PointCloud<pcl::PointXYZI>::Ptr accumulated_cloud(new pcl::PointCloud<pcl::PointXYZI>);
// pcl::visualization::CloudViewer viewer("viewer");

void GroundCloudCallback(const sensor_msgs::PointCloud2ConstPtr& cloud_msg) {
    // 将ROS消息转换为PCL点云格式

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::PointCloud<pcl::PointXYZ>::Ptr ground_cloud(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::fromROSMsg(*cloud_msg, *cloud);

    // 分割地面点云
    if (ground_extractor.ExtractGroundCloud(cloud, ground_cloud, 500, 0.1)) {
        ROS_INFO("Ground plane extraction successful.");
    } else {
        ROS_WARN("Ground plane extraction failed.");
    }
    // 求yaw
    double front_x = 2.275075, front_y = 0.047439;
    double rear_x = 2.370587, rear_y = 0.058277;
    double yaw = ground_extractor.GetYaw(front_x, front_y, rear_x, rear_y); // 数模+测量计算得到

    // 求pitch和roll
    const GroundExtractor::PlaneCoefficients& ground_coefficients = ground_extractor.getGroundCoefficients();
    Eigen::Vector3d plane_normal(ground_coefficients.a, ground_coefficients.b, ground_coefficients.c);
    Eigen::Vector3d master_z(0, 0, 1);

    Eigen::Vector3d rot_axis = plane_normal.cross(master_z);       // rotation vector n
    rot_axis.normalize();
    double alpha = -std::acos(plane_normal.dot(master_z));         // rotation angle a
    Eigen::Matrix3d R_mp;
    R_mp = Eigen::AngleAxisd(alpha, rot_axis);  // Z1Y2X3旋转矩阵 -> 内旋
    std::cout << "R1:\n" << R_mp << std::endl;

    Eigen::Vector3d t_mp(0, 0, -ground_coefficients.intercept / plane_normal(2));
    Eigen::Matrix4d T_pm = ground_extractor.GetMatrix(t_mp, R_mp).inverse();

    double roll = ground_extractor.GetRoll(T_pm);
    double pitch = ground_extractor.GetPitch(T_pm);
    double height = ground_coefficients.intercept / ground_coefficients.c;
    std::cout << "roll = " << roll * 180 / M_PI << "  pitch = " << pitch * 180 / M_PI << std::endl;
    std::cout << "height = " << height << std::endl;

    // lidar2vehicle
    Eigen::Matrix4f transform_yaw = Eigen::Matrix4f::Identity(); // 绕Z轴
    float theta_yaw = yaw * M_PI / 180.0; 
    transform_yaw(0, 0) = cos(theta_yaw);
    transform_yaw(0, 1) = -sin(theta_yaw);
    transform_yaw(1, 0) = sin(theta_yaw);
    transform_yaw(1, 1) = cos(theta_yaw);
    // transform_yaw(0, 0) = cos(theta_yaw);
    // transform_yaw(2, 1) = -sin(theta_yaw);
    // transform_yaw(2, 1) = sin(theta_yaw);
    // transform_yaw(2, 2) = cos(theta_yaw);
    Eigen::Matrix4f transform_roll = Eigen::Matrix4f::Identity();  // 绕X轴
    float theta_roll = roll;
    transform_roll(1, 1) = cos(theta_roll);
    transform_roll(1, 2) = -sin(theta_roll);
    transform_roll(2, 1) = sin(theta_roll);
    transform_roll(2, 2) = cos(theta_roll);

    Eigen::Matrix4f transform_pitch = Eigen::Matrix4f::Identity(); // 绕Y轴
    float theta_pitch = pitch;
    transform_pitch(0, 0) = cos(theta_pitch);
    transform_pitch(0, 2) = sin(theta_pitch);
    transform_pitch(2, 0) = -sin(theta_pitch);
    transform_pitch(2, 2) = cos(theta_pitch);

    // // ZXY旋转矩阵 
    // Eigen::Matrix4f transform_rotation = transform_pitch * transform_roll * transform_yaw;
    // ZYX旋转矩阵
    Eigen::Matrix4f transform_rotation = transform_roll * transform_pitch * transform_yaw;
    std::cout << "transform_rotation: \n" << transform_rotation << std::endl;

    // 平移 T (x, y, z)
    std::vector<double> translate = {0, 0, 0};
    Eigen::Matrix4f transform = transform_rotation;
    transform(0, 3) = 1.351; // x 方向平移
    transform(1, 3) = -0.01; // y 方向平移
    transform(2, 3) = 1.328; // z 方向平移

    pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_vehicle_cloud(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::transformPointCloud(*ground_cloud, *transformed_vehicle_cloud, transform);

    ros::NodeHandle nh_ground;
    // 发布未转换的地面点云数据
    static ros::Publisher ground_pub = nh_ground.advertise<sensor_msgs::PointCloud2>("raw_ground_cloud", 10);
    sensor_msgs::PointCloud2 ground_output;
    pcl::toROSMsg(*ground_cloud, ground_output);
    ground_output.header.stamp = cloud_msg->header.stamp;
    ground_output.header.frame_id = "Pandar40P";
    ground_pub.publish(ground_output);
    
    // 发布坐标转换后的地面点云数据
    static ros::Publisher transformed_ground_pub = nh_ground.advertise<sensor_msgs::PointCloud2>("transformed_ground_cloud", 10);
    sensor_msgs::PointCloud2 point_output;
    pcl::toROSMsg(*transformed_vehicle_cloud, point_output);
    point_output.header.stamp = cloud_msg->header.stamp;
    point_output.header.frame_id = "Pandar40P";
    transformed_ground_pub.publish(point_output);
}

void PlateCloudCallback(const sensor_msgs::PointCloud2ConstPtr& cloud_msg) {
    // 将ROS消息转换为PCL点云格式
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>);
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_plane(new pcl::PointCloud<pcl::PointXYZI>());

    pcl::fromROSMsg(*cloud_msg, *cloud);

    // 点云滤波
    plate_extractor.cloudPassThrough(cloud, "y", -2, 5);
    plate_extractor.cloudPassThrough(cloud, "x", 3.5, 5.5);
    plate_extractor.cloudPassThrough(cloud, "z", -0.8, 1.0);

    // TODO：输出滤波后的点云数据
    // viewer.showCloud(cloud);
    // sleep(2);

    // 去除离群点
    plate_extractor.cloudStatisticalOutlierRemoval(cloud);
    // TODO：输出去除离群点后的点云数据
    // viewer.showCloud(cloud);
    // sleep(2);

    *accumulated_cloud += *cloud;
    frame_count ++;
    if (frame_count >= max_frames) {
        frame_count = 0;
        pcl::io::savePCDFileASCII("accumulated_cloud.pcd", *accumulated_cloud);
        std::cout << "save accumulated_cloud.pcd success !!" << std::endl;

        // 平面及边界提取
        plate_extractor.Plane_fitting(accumulated_cloud, cloud_plane);
        // plate_extractor.cloudEdge(plate_cloud);

        // 读取计算得到的平面角点
        const std::vector<cv::Point2f>& coner_points = plate_extractor.getConerPoints();
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr conerpoint_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
        for (auto i = 0; i < coner_points.size(); i++) {
            cv::Point2f coner_point = coner_points[i];
            plate_extractor.SpatialPoint(coner_point, conerpoint_cloud);
        }

        // std::vector<Eigen::Vector3d> p1, p2;
        // pcl::PointXYZ pointA, pointB, pointC, pointD;
        // cv::Point2f coner_point_A = coner_points[0];
        // cv::Point2f coner_point_B = coner_points[1];
        // cv::Point2f coner_point_C = coner_points[2];
        // cv::Point2f coner_point_D = coner_points[3];
        // plate_extractor.SpatialPoint(coner_point_A, conerpoint_cloud, pointA);
        // plate_extractor.SpatialPoint(coner_point_B, conerpoint_cloud, pointB);
        // plate_extractor.SpatialPoint(coner_point_C, conerpoint_cloud, pointC);
        // plate_extractor.SpatialPoint(coner_point_D, conerpoint_cloud, pointD);

        // Eigen::Vector3d pointA_eigen(pointA.x, pointA.y, pointA.z);
        // Eigen::Vector3d pointB_eigen(pointB.x, pointB.y, pointB.z);
        // Eigen::Vector3d pointC_eigen(pointC.x, pointC.y, pointC.z);
        // Eigen::Vector3d pointD_eigen(pointD.x, pointD.y, pointD.z);

        // p1.push_back(pointA_eigen);
        // p1.push_back(pointB_eigen);
        // p1.push_back(pointC_eigen);
        // p1.push_back(pointD_eigen);

        // Eigen::Vector3d pointA_xoy(0, 0, 0);
        // Eigen::Vector3d pointB_xoy(0.76, 0, 0);
        // Eigen::Vector3d pointC_xoy(0.76, 0.58, 0);
        // Eigen::Vector3d pointD_xoy(0, 0.58, 0);
        // p2.push_back(pointA_xoy);
        // p2.push_back(pointB_xoy);
        // p2.push_back(pointC_xoy);
        // p2.push_back(pointD_xoy);

        // Eigen::Matrix3d R_12;
        // Eigen::Vector3d t_12;

        // plate_extractor.bundleAdjustment(p1, p2, R_12, t_12);
        // for(std::size_t i = 0; i < p2.size(); i ++) {
        //     Eigen::Vector3d p = R_12 * p2[i] + t_12;
        //     pcl::PointXYZRGB thisColor;
        //     thisColor.x = p.x();
        //     thisColor.y = p.y();
        //     thisColor.z = p.z();
        //     thisColor.r = 255;
        //     thisColor.g = 0;
        //     thisColor.b = 0;
        //     conerpoint_cloud->push_back(thisColor);
        // }
        // std::cout << "result R_12:\n" << R_12 << std::endl;
        // std::cout << "result t_12:\n" << t_12.transpose() << std::endl;


        // 读取计算得到的棋盘格角点
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr chesspoint_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
        const std::vector<cv::Point2f>& chess_points = plate_extractor.getChessPoints();
        for (auto i = 0; i < chess_points.size(); i++) {
            cv::Point2f chess_point = chess_points[i];
            plate_extractor.SpatialPoint(chess_point, chesspoint_cloud);
        }

        // 发布
        ros::NodeHandle nh_plate;
        static ros::Publisher coner_pub = nh_plate.advertise<sensor_msgs::PointCloud2>("coner_cloud", 10);
        sensor_msgs::PointCloud2 coner_output;
        pcl::toROSMsg(*conerpoint_cloud, coner_output);
        coner_output.header.stamp = cloud_msg->header.stamp;
        coner_output.header.frame_id = "Pandar40P";
        coner_pub.publish(coner_output);

        static ros::Publisher chessboard_pub = nh_plate.advertise<sensor_msgs::PointCloud2>("chessboard_cloud", 10);
        sensor_msgs::PointCloud2 chessboard_output;
        pcl::toROSMsg(*chesspoint_cloud, chessboard_output);
        chessboard_output.header.stamp = cloud_msg->header.stamp;
        chessboard_output.header.frame_id = "Pandar40P";
        chessboard_pub.publish(chessboard_output);

        pcl::PointCloud<pcl::PointXYZI>::Ptr plate_plane_ = plate_extractor.getPlatePlane();
        static ros::Publisher plate_pub = nh_plate.advertise<sensor_msgs::PointCloud2>("plate_plane", 10);
        sensor_msgs::PointCloud2 plate_output;
        pcl::toROSMsg(*plate_plane_, plate_output);
        plate_output.header.stamp = cloud_msg->header.stamp;
        plate_output.header.frame_id = "Pandar40P";
        plate_pub.publish(plate_output);

        accumulated_cloud->clear();
    }
}

int main(int argc, char** argv) {
    ros::init(argc, argv, "hesailidar_calib_node");
    ros::NodeHandle nh;

    ros::Subscriber groundcloud_sub = nh.subscribe("/hesai/pandar", 10, GroundCloudCallback);
    ros::Subscriber plate_sub = nh.subscribe("/hesai/pandar", 10, PlateCloudCallback);
    ros::spin();
    return 0;
}
