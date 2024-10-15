#include "GroundExtract.h"

GroundExtractor::GroundExtractor() {
    ground_coefficients_ = {0.0f, 0.0f, 0.0f, 0.0f};
}

void GroundExtractor::cloudPassThrough(const pcl::PointCloud<pcl::PointXYZ>::Ptr& input_cloud,  
                                       const std::string& axis, float axis_min, float axis_max) {
    pcl::PassThrough<pcl::PointXYZ> passthrough;
    passthrough.setInputCloud(input_cloud);
    passthrough.setFilterFieldName(axis);
    passthrough.setFilterLimits(axis_min, axis_max);
    passthrough.filter(*input_cloud);
}

double GroundExtractor::GetYaw(double front_x, double front_y, double rear_x, double rear_y) {
    double yaw = (atan2(front_y, front_x) + atan2(rear_y, rear_x)) / 2;
    return yaw;
}

Eigen::Matrix4d GroundExtractor::GetMatrix(const Eigen::Vector3d& translation,
                                           const Eigen::Matrix3d& rotation) {
    Eigen::Matrix4d ret = Eigen::Matrix4d::Identity();
    ret.block<3, 1>(0, 3) = translation;
    ret.block<3, 3>(0, 0) = rotation;
    return ret;
}

double GroundExtractor::GetRoll(const Eigen::Matrix4d& matrix) {
    Eigen::Matrix3d R = matrix.block<3, 3>(0, 0);
    // Eigen::Vector3d Rx = R.col(0);
    Eigen::Vector3d Ry = R.col(1);
    Eigen::Vector3d Rz = R.col(2);
    // double yaw = atan2(Rx(1), Rx(0));
    // double roll = atan2(Rz(0) * sin(yaw) - Rz(1) * cos(yaw), -Ry(0) * sin(yaw) + Ry(1) * cos(yaw));
    double roll = atan2(Ry(2), Rz(2));
    return roll;
}

double GroundExtractor::GetPitch(const Eigen::Matrix4d& matrix) {
    Eigen::Matrix3d R = matrix.block<3, 3>(0, 0);
    Eigen::Vector3d Rx = R.col(0);
    double yaw = atan2(Rx(1), Rx(0));
    double pitch = atan2(-Rx(2), Rx(0) * cos(yaw) + Rx(1) * sin(yaw));
    return pitch;
}

bool GroundExtractor::ExtractGroundCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, 
                                         pcl::PointCloud<pcl::PointXYZ>::Ptr& ground_cloud, 
                                         int max_iterations, float distance_threshold) {
    cloudPassThrough(cloud, "z", -1.7, -1.3);

    // 点云分割
    pcl::SACSegmentation<pcl::PointXYZ> seg;
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
    pcl::ExtractIndices<pcl::PointXYZ> extract;

    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setAxis(Eigen::Vector3f(0, 0, 1)); 
    seg.setMaxIterations(max_iterations);       
    seg.setDistanceThreshold(distance_threshold); 

    seg.setInputCloud(cloud);
    seg.segment(*inliers, *coefficients);
    if (inliers->indices.size() == 0) {
        ROS_WARN("Could not estimate a planar model for the given dataset.");
        return false;
    }

    extract.setInputCloud(cloud);
    extract.setIndices(inliers);
    extract.setNegative(false);
    extract.filter(*ground_cloud);

    ground_coefficients_.a = coefficients->values[0];
    ground_coefficients_.b = coefficients->values[1];
    ground_coefficients_.c = coefficients->values[2];
    ground_coefficients_.intercept = coefficients->values[3];
    return true;
}