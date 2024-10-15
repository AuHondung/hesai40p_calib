#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>

#include <pcl_conversions/pcl_conversions.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/common/transforms.h>
#include <pcl/common/common.h>
#include <pcl/filters/passthrough.h>

#include <Eigen/Dense>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <tf2_eigen/tf2_eigen.h>
#include <geometry_msgs/TransformStamped.h>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

// #include <pcl/surface/convex_hull.h>
// #include <pcl/surface/mls.h>
// #include "prcal.h"

typedef pcl::PointCloud<pcl::PointXYZ> PointCloud;

struct PlaneCoefficients {
    float a;
    float b;
    float c;
    float intercept;
};

PlaneCoefficients plane = {0.0, 0.0, 0.0, 0.0};

void cloudPassThrough(const PointCloud::Ptr& input_cloud, const PointCloud::Ptr& filtered_cloud, 
                        const std::string& axis, float axis_min, float axis_max) {
    pcl::PassThrough<pcl::PointXYZ> passthrough;
    passthrough.setInputCloud(input_cloud);
    passthrough.setFilterFieldName(axis);
    passthrough.setFilterLimits(axis_min, axis_max);
    passthrough.filter(*filtered_cloud);
}

bool Collinear(pcl::PointXYZ A, pcl::PointXYZ C, pcl::PointXYZ B) {
    // 三角形面积*2=叉积的模|axb|=a*b*sin(theta)
    float SABC = sqrt(((B.x - A.x)*(B.y - C.y) - (B.x - C.x)*(B.y - A.y)) * ((B.x - A.x)*(B.y - C.y) - (B.x - C.x)*(B.y - A.y))
                    + ((B.x - A.x)*(B.z - C.z) - (B.x - C.x)*(B.z - A.z)) * ((B.x - A.x)*(B.z - C.z) - (B.x - C.x)*(B.z - A.z))
                    + ((B.y - A.y)*(B.z - C.z) - (B.y - C.y)*(B.z - A.z)) * ((B.y - A.y)*(B.z - C.z) - (B.y - C.y)*(B.z - A.z)));
    // 底边边长
    float lAC = sqrt((A.x - C.x)*(A.x - C.x) + (A.y - C.y)*(A.y - C.y) + (A.z - C.z)*(A.z - C.z));
    // 点到直线的距离
    float ld = SABC / lAC;
    // std::cout << "ld = "<< ld << "ac:" << lAC << std::endl;
    if(ld < 0.06) return false;
    return true;
}

// 二维直线拟合
void LineFitLeastFit(const std::vector<cv::Point2f> &_points, float & _k, float & _b, float & _r) {
    // https://blog.csdn.net/jjjstephen/article/details/108053148?utm_medium=distribute.pc_relevant.none-task-blog-2~default~baidujs_title~default-0.pc_relevant_default&spm=1001.2101.3001.4242.1&utm_relevant_index=3
    float B = 0.0f;
    float A = 0.0f;
    float D = 0.0f;
    float C = 0.0f;

    int N = _points.size();
    for (int i = 0; i < N; i++) {
            B += _points[i].x;
            A += _points[i].x * _points[i].x;
            D += _points[i].y;
            C += _points[i].x * _points[i].y;
    }
    if ((N * A - B * B) == 0) return;
    _k = (N * C - B * D) / (N * A - B * B);
    _b = (A * D - C * B) / (N * A - B * B);
    // 计算相关系数
    float Xmean = B / N;
    float Ymean = D / N;

    float tempX = 0.0f;
    float tempY = 0.0f;
    float rDenominator = 0.0;
    for (int i = 0; i < N; i++) {
            tempX += (_points[i].x - Xmean) * (_points[i].x - Xmean);
            tempY += (_points[i].y - Ymean) * (_points[i].y - Ymean);
            rDenominator += (_points[i].x - Xmean) * (_points[i].y - Ymean);
    }

    float SigmaXY = sqrt(tempX) * sqrt(tempY);
    if (SigmaXY == 0) return;
    _r = rDenominator / SigmaXY;
}

// 求平面两线交点
void intersection(cv::Point2f &point, float &A_k, float &A_b, float &B_k, float &B_b) {
    Eigen::Matrix<double, 2, 2> A;
    A(0, 0) = 1;
    A(0, 1) = -A_k;
    A(1, 0) = 1;
    A(1, 1) = -B_k;
    Eigen::Matrix<double, 2, 1> B;
    B(0,0) = A_b;
    B(1,0) = B_b;

    Eigen::Matrix<double, 2, 1> xy = A.fullPivHouseholderQr().solve(B);
    point.x = xy(1,0);
    point.y = xy(0,0);
}

// 空间点估计
void SpatialPoint(cv::Point2f &point2D, pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_all, PlaneCoefficients plate_coefficients) {
    pcl::PointXYZRGB thisColor;
    thisColor.x = -(plate_coefficients.b * point2D.x + plate_coefficients.c * point2D.y + plate_coefficients.intercept) / plate_coefficients.a;
    thisColor.y = point2D.x;
    thisColor.z = point2D.y;
    thisColor.r = 0;
    thisColor.g = 255;
    thisColor.b = 0;
    cloud_all->push_back(thisColor);
}

// 添加空间直线上的点
void addSpatialPoint(cv::Point2f &point2A, cv::Point2f &point2B, float & _k, float & _b, 
                    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_all, PlaneCoefficients plate_coefficients) {
    float step = 0.01;
    if (point2A.x < point2B.x) {
        for (float a = point2A.x; a < point2B.x; a = a + step) {
            cv::Point2f thisPoint;
            thisPoint.x = a;
            thisPoint.y = _k * a + _b;
            SpatialPoint(thisPoint, cloud_all, plate_coefficients);
        }
    } else {
        for (float a = point2B.x; a < point2A.x; a = a + step) {
            cv::Point2f thisPoint;
            thisPoint.x = a;
            thisPoint.y = _k * a + _b;
            SpatialPoint(thisPoint, cloud_all, plate_coefficients);
        }
    }
}

static Eigen::Matrix4d GetMatrix(const Eigen::Vector3d& translation,
                                    const Eigen::Matrix3d& rotation) {
    Eigen::Matrix4d ret = Eigen::Matrix4d::Identity();
    ret.block<3, 1>(0, 3) = translation;
    ret.block<3, 3>(0, 0) = rotation;
    return ret;
}

static double GetRoll(const Eigen::Matrix4d& matrix) {
    Eigen::Matrix3d R = matrix.block<3, 3>(0, 0);
    // Eigen::Vector3d Rx = R.col(0);
    Eigen::Vector3d Ry = R.col(1);
    Eigen::Vector3d Rz = R.col(2);
    // double yaw = atan2(Rx(1), Rx(0));
    // double roll = atan2(Rz(0) * sin(yaw) - Rz(1) * cos(yaw), -Ry(0) * sin(yaw) + Ry(1) * cos(yaw));
    double roll = atan2(Ry(2), Rz(2));
    return roll;
}

static double GetPitch(const Eigen::Matrix4d& matrix) {
    Eigen::Matrix3d R = matrix.block<3, 3>(0, 0);
    Eigen::Vector3d Rx = R.col(0);
    double yaw = atan2(Rx(1), Rx(0));
    double pitch = atan2(-Rx(2), Rx(0) * cos(yaw) + Rx(1) * sin(yaw));
    return pitch;
}

bool ExtractGroundCloud(PointCloud::Ptr& cloud, PointCloud::Ptr& ground_cloud, PlaneCoefficients& ground_coefficients,
                        int max_iterations, float distance_threshold) {
    pcl::PassThrough<pcl::PointXYZ> pass;
    PointCloud::Ptr cloud_filtered_z(new PointCloud());

    cloudPassThrough(cloud, cloud_filtered_z, "z", -1.7, -1.3);

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

    seg.setInputCloud(cloud_filtered_z);
    seg.segment(*inliers, *coefficients);
    if (inliers->indices.size() == 0) {
        ROS_WARN("Could not estimate a planar model for the given dataset.");
        return false;
    }

    extract.setInputCloud(cloud_filtered_z);
    extract.setIndices(inliers);
    extract.setNegative(false);
    extract.filter(*ground_cloud);

    ground_coefficients.a = coefficients->values[0];
    ground_coefficients.b = coefficients->values[1];
    ground_coefficients.c = coefficients->values[2];
    ground_coefficients.intercept = coefficients->values[3];
    return true;
}

bool ExtractPlateCloud(PointCloud::Ptr& cloud, PointCloud::Ptr& plate_cloud, PlaneCoefficients& plate_coefficients,
                       int max_iterations, float distance_threshold) {
    // 点云过滤
    PointCloud::Ptr cloud_filtered_x(new PointCloud());
    PointCloud::Ptr cloud_filtered(new PointCloud());
    cloudPassThrough(cloud, cloud_filtered_x, "x", 3.0, 5.0);
    cloudPassThrough(cloud_filtered_x, cloud_filtered, "y", -1.0, 1.0);

    // 标定版点云分割
    pcl::SACSegmentation<pcl::PointXYZ> seg;
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
    
    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setMaxIterations(max_iterations);            
    seg.setDistanceThreshold(distance_threshold);    
    seg.setInputCloud(cloud_filtered);
    seg.segment(*inliers, *coefficients);
    if (inliers->indices.size() == 0) {
        ROS_WARN("Could not estimate a planar model for the given dataset.");
        return false;
    }
    pcl::ExtractIndices<pcl::PointXYZ> extract;
    extract.setInputCloud(cloud_filtered);
    extract.setIndices(inliers);
    extract.setNegative(false);
    extract.filter(*plate_cloud);

    plate_coefficients.a = coefficients->values[0];
    plate_coefficients.b = coefficients->values[1];
    plate_coefficients.c = coefficients->values[2];
    plate_coefficients.intercept = coefficients->values[3];
    return true;
}

void ExtractEdgeLines(PointCloud::Ptr& plate_cloud, std::vector<PointCloud::Ptr>& edge_lines) {
    PointCloud::Ptr remaining_cloud(new PointCloud(*plate_cloud));
    pcl::SACSegmentation<pcl::PointXYZ> seg;
    pcl::ExtractIndices<pcl::PointXYZ> extract;
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);

    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_LINE);       // 拟合模型类型为直线
    seg.setMethodType(pcl::SAC_RANSAC);         // 使用RANSAC算法
    seg.setMaxIterations(500);                 // 最大迭代次数
    seg.setDistanceThreshold(0.01);             // 点到模型的距离阈值

    int num_lines = 4; // 期望找到4条边界线
    for (int i = 0; i < num_lines; ++i) {
        seg.setInputCloud(remaining_cloud);
        seg.segment(*inliers, *coefficients);

        if (inliers->indices.size() == 0) {
            ROS_WARN("Could not estimate a line model for the given dataset.");
            break;
        }

        // 提取边界线点云
        pcl::PointCloud<pcl::PointXYZ>::Ptr edge_line(new pcl::PointCloud<pcl::PointXYZ>);
        extract.setInputCloud(remaining_cloud);
        extract.setIndices(inliers);
        extract.setNegative(false);
        extract.filter(*edge_line);
        edge_lines.push_back(edge_line);

        // 去除已经提取的直线的点云，以便在下一次循环中拟合出新的直线
        extract.setNegative(true);
        extract.filter(*remaining_cloud);
    }
}

void ExtractEdgeLines_v2(PointCloud::Ptr& plate_cloud, std::vector<PointCloud::Ptr>& boundary_lines) {
    // 计算边界
    pcl::PointXYZ minPt, maxPt;
    pcl::getMinMax3D(*plate_cloud, minPt, maxPt);

    pcl::PointXYZ corner1(minPt.x, minPt.y, minPt.z);
    pcl::PointXYZ corner2(maxPt.x, minPt.y, minPt.z);
    pcl::PointXYZ corner3(maxPt.x, maxPt.y, minPt.z);
    pcl::PointXYZ corner4(minPt.x, maxPt.y, minPt.z);

    // 提取四条边界线上的点云
    // std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> boundary_lines(4);
    for (int i = 0; i < 4; ++i) {
        boundary_lines[i] = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>);
    }

    for (const auto& point : plate_cloud->points) {
        if (std::abs(point.z - minPt.z) < 0.05) {
            boundary_lines[0]->points.push_back(point);  // 下边界线
        } else if (std::abs(point.z - maxPt.z) < 0.05) {
            boundary_lines[1]->points.push_back(point);  // 上边界线
        } else if (std::abs(point.x - maxPt.x) < 0.01) {
            boundary_lines[2]->points.push_back(point);  // 左边界线
        } else if (std::abs(point.x - maxPt.x) < 0.01) {
            boundary_lines[3]->points.push_back(point);  // 右边界线
        }
    }
}

void cloudEdge(PointCloud::Ptr& plate_cloud, PlaneCoefficients& plate_coefficients, std::vector<PointCloud::Ptr>& boundary_lines) {
    int index[40][2];
    memset(index, -1, sizeof(index)); // 赋值
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_all(new pcl::PointCloud<pcl::PointXYZRGB>);
    for(std::size_t i = 0; i < plate_cloud->size(); i++) {
        // 计算垂直俯仰角
        float angle = atan(plate_cloud->points[i].z / sqrt( pow(plate_cloud->points[i].x, 2) +  pow(plate_cloud->points[i].y, 2))) * 180 / M_PI;
        int scanID = 0;  // lidar扫描线ID
        scanID = int((angle + 25) * 3 + 0.5);  // +0.5 用于四舍五入
        if(0 <= scanID && scanID < 1000) {
            if(index[scanID][0] == -1)
                index[scanID][0] = i;
            else
                index[scanID][1] = i;
        }
        pcl::PointXYZRGB thisColor;
        thisColor.x = plate_cloud->points[i].x;
        thisColor.y = plate_cloud->points[i].y;
        thisColor.z = plate_cloud->points[i].z;
        thisColor.r = 255;
        thisColor.g = 255;
        thisColor.b = 255;
        cloud_all->push_back(thisColor);
    }
    // 提取边缘
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_edge(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_edge_left(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_edge_right(new pcl::PointCloud<pcl::PointXYZ>);
    for(int i = 0; i < 1000; i ++) {
        if(index[i][0] != -1) {
            pcl::PointXYZRGB thisColor;
            thisColor.x = plate_cloud->points[index[i][0]].x;
            thisColor.y = plate_cloud->points[index[i][0]].y;
            thisColor.z = plate_cloud->points[index[i][0]].z;
            thisColor.r = 255;
            thisColor.g = 255;
            thisColor.b = 255;
            // cloud_all->push_back(thisColor);
            cloud_edge->push_back(thisColor);
            cloud_edge_left->push_back(plate_cloud->points[index[i][0]]);
        }
    }
    for(int i = 999; i >= 0; i--) {
        if(index[i][1] != -1) {
            pcl::PointXYZRGB thisColor;
            thisColor.x = plate_cloud->points[index[i][1]].x;
            thisColor.y = plate_cloud->points[index[i][1]].y;
            thisColor.z = plate_cloud->points[index[i][1]].z;
            thisColor.r = 255;
            thisColor.g = 255;
            thisColor.b = 255;
            // cloud_all->push_back(thisColor);
            cloud_edge->push_back(thisColor);
            cloud_edge_right->push_back(plate_cloud->points[index[i][1]]);
        }
    }
    // 划分4条线
    int d_a = 0;
    int b_c = 0;
    std::vector<cv::Point2f> pointsA;
    std::vector<cv::Point2f> pointsB;
    std::vector<cv::Point2f> pointsC;
    std::vector<cv::Point2f> pointsD;

    for (std::size_t i = 0; i < cloud_edge_left->size() - 2; i ++) {
        // 左侧第三个点到前两个点的距离
        if (Collinear(cloud_edge_left->points[i], cloud_edge_left->points[i + 1], cloud_edge_left->points[i + 2])) {
            d_a = i + 1;
            break;
        }
    }

    for (std::size_t i = 0; i < cloud_edge_right->size() - 2; i ++) {
        // 右侧第三个点到前两个点的距离
        if (Collinear(cloud_edge_right->points[i], cloud_edge_right->points[i + 1], cloud_edge_right->points[i + 2])) {
            b_c = i + 1;
            break;
        }
    }

    for (std::size_t i = 0; i < cloud_edge_left->size(); i ++) {
        if (i < d_a) {
            cv::Point2f thisPoint;
            thisPoint.x = cloud_edge_left->points[i].y;
            thisPoint.y = cloud_edge_left->points[i].z;
            pointsD.push_back(thisPoint);
        } else {
            cv::Point2f thisPoint;
            thisPoint.x = cloud_edge_left->points[i].y;
            thisPoint.y = cloud_edge_left->points[i].z;
            pointsA.push_back(thisPoint);
        }
    }

    for (std::size_t i = 0; i < cloud_edge_right->size(); i ++) {
        if (i < b_c) {
            cv::Point2f thisPoint;
            thisPoint.x = cloud_edge_right->points[i].y;
            thisPoint.y = cloud_edge_right->points[i].z;
            pointsB.push_back(thisPoint);
        } else {
            cv::Point2f thisPoint;
            thisPoint.x = cloud_edge_right->points[i].y;
            thisPoint.y = cloud_edge_right->points[i].z;
            pointsC.push_back(thisPoint);
        }
    }
    
    float A_k, A_b, A_r;
    float B_k, B_b, B_r;
    float C_k, C_b, C_r;
    float D_k, D_b, D_r;

    LineFitLeastFit(pointsA, A_k, A_b, A_r);
    LineFitLeastFit(pointsB, B_k, B_b, B_r);
    LineFitLeastFit(pointsC, C_k, C_b, C_r);
    LineFitLeastFit(pointsD, D_k, D_b, D_r);

    // 求yoz平面交点
    cv::Point2f pointAb;
    cv::Point2f pointBc;
    cv::Point2f pointCd;
    cv::Point2f pointDa;
    intersection(pointAb, A_k, A_b, B_k, B_b);
    intersection(pointBc, B_k, B_b, C_k, C_b);
    intersection(pointCd, C_k, C_b, D_k, D_b);
    intersection(pointDa, D_k, D_b, A_k, A_b);

    // 求空间点 
    SpatialPoint(pointAb, cloud_all, plate_coefficients);
    SpatialPoint(pointBc, cloud_all, plate_coefficients);
    SpatialPoint(pointCd, cloud_all, plate_coefficients);
    SpatialPoint(pointDa, cloud_all, plate_coefficients);

    // 增加空间点
    addSpatialPoint(pointAb, pointBc, B_k, B_b, cloud_all, plate_coefficients);
    addSpatialPoint(pointBc, pointCd, C_k, C_b, cloud_all, plate_coefficients);
    addSpatialPoint(pointCd, pointDa, D_k, D_b, cloud_all, plate_coefficients);
    addSpatialPoint(pointDa, pointAb, A_k, A_b, cloud_all, plate_coefficients);
}

void GroundCloudCallback(const sensor_msgs::PointCloud2ConstPtr& cloud_msg) {
    // 将ROS消息转换为PCL点云格式
    PointCloud::Ptr cloud(new PointCloud);
    PointCloud::Ptr ground_cloud(new PointCloud());
    PlaneCoefficients ground_coefficients;
    pcl::fromROSMsg(*cloud_msg, *cloud);

    if (ExtractGroundCloud(cloud, ground_cloud, ground_coefficients, 500, 0.1)) {
        // ROS_INFO("Ground plane extraction successful.");
        ROS_INFO("Plane equation: %fx + %fy + %fz + %f = 0", 
        ground_coefficients.a, 
        ground_coefficients.b, 
        ground_coefficients.c, 
        ground_coefficients.intercept);
    } else {
        ROS_WARN("Ground plane extraction failed.");
    }
    // 求pitch和roll
    Eigen::Vector3d plane_normal(ground_coefficients.a, ground_coefficients.b, ground_coefficients.c);
    Eigen::Vector3d master_z(0, 0, 1);
    // rotation vector n
    Eigen::Vector3d rot_axis = plane_normal.cross(master_z);
    rot_axis.normalize();
    // rotation angle a
    double alpha = -std::acos(plane_normal.dot(master_z));
    Eigen::Matrix3d R_mp;
    R_mp = Eigen::AngleAxisd(alpha, rot_axis);
    Eigen::Vector3d t_mp(0, 0, -ground_coefficients.intercept / plane_normal(2));
    Eigen::Matrix4d T_pm = GetMatrix(t_mp, R_mp).inverse();

    double roll = GetRoll(T_pm);
    double pitch = GetPitch(T_pm);
    std::cout << "roll = " << roll << "  pitch = " << pitch << std::endl;

    double yaw = 1.3; // 数模+测量计算得到
    // 1. lidar2vehicle
    Eigen::Matrix4f transform_vehicle = Eigen::Matrix4f::Identity();
    float theta_vehicle = yaw * M_PI / 180.0; 
    transform_vehicle(0, 0) = cos(theta_vehicle);
    transform_vehicle(0, 1) = -sin(theta_vehicle);
    transform_vehicle(1, 0) = sin(theta_vehicle);
    transform_vehicle(1, 1) = cos(theta_vehicle);

    Eigen::Matrix4f transform_roll = Eigen::Matrix4f::Identity();
    float theta_roll = roll;
    transform_roll(1, 1) = cos(theta_roll);
    transform_roll(1, 2) = -sin(theta_roll);
    transform_roll(2, 1) = sin(theta_roll);
    transform_roll(2, 2) = cos(theta_roll);

    Eigen::Matrix4f transform_pitch = Eigen::Matrix4f::Identity();
    float theta_pitch = pitch;
    transform_pitch(0, 0) = cos(theta_pitch);
    transform_pitch(0, 2) = sin(theta_pitch);
    transform_pitch(2, 0) = -sin(theta_pitch);
    transform_pitch(2, 2) = cos(theta_pitch);

    Eigen::Matrix4f transform_rotation = transform_vehicle * transform_pitch * transform_roll;
    // 平移 T (x, y, z)
    Eigen::Matrix4f transform = transform_rotation;
    transform(0, 3) = 1.351; // x 方向平移
    transform(1, 3) = -0.01; // y 方向平移
    transform(2, 3) = 1.328; // z 方向平移

    PointCloud::Ptr transformed_vehicle_cloud(new PointCloud);
    pcl::transformPointCloud(*ground_cloud, *transformed_vehicle_cloud, transform);

    // // 2. 坐标系对齐
    // Eigen::Matrix4f transform = Eigen::Matrix4f::Identity();
    // float theta = 90 * M_PI / 180; 
    // transform(0, 0) = cos(theta);
    // transform(0, 1) = -sin(theta);
    // transform(1, 0) = sin(theta);
    // transform(1, 1) = cos(theta);
    // // 平移 T (x, y, z)
    // transform(0, 3) = 0; // x 方向平移 0
    // transform(1, 3) = 0; // y 方向平移 0
    // transform(2, 3) = 0; // z 方向平移 0
    // PointCloud::Ptr transformed_cloud(new PointCloud);
    // pcl::transformPointCloud(*transformed_vehicle_cloud, *transformed_cloud, transform);

    // // 3. vehicle2gps
    // Eigen::Matrix4f transform_gps = Eigen::Matrix4f::Identity();
    // float gps_yaw = 45;
    // // 旋转 R
    // float theta_gps = gps_yaw * M_PI / 180; 
    // transform_gps(0, 0) = cos(theta_gps);
    // transform_gps(0, 1) = -sin(theta_gps);
    // transform_gps(1, 0) = sin(theta_gps);
    // transform_gps(1, 1) = cos(theta_gps);
    // // 平移 T (x, y, z)
    // transform_gps(0, 3) = 1.0; // x 方向平移
    // transform_gps(1, 3) = 2.0; // y 方向平移
    // transform_gps(2, 3) = 3.0; // z 方向平移
    // // 对地面点云进行坐标系变换
    // PointCloud::Ptr transformed_gps_cloud(new PointCloud);
    // pcl::transformPointCloud(*transformed_cloud, *transformed_gps_cloud, transform_gps);

    // 发布坐标转换后的地面点云数据
    static ros::Publisher ground_pub = ros::NodeHandle().advertise<sensor_msgs::PointCloud2>("raw_ground_cloud", 10);
    sensor_msgs::PointCloud2 ground_output;
    pcl::toROSMsg(*ground_cloud, ground_output);
    ground_output.header.stamp = cloud_msg->header.stamp;
    ground_output.header.frame_id = "Pandar40P";
    ground_pub.publish(ground_output);

    static ros::Publisher pub = ros::NodeHandle().advertise<sensor_msgs::PointCloud2>("transformed_ground_cloud", 10);
    sensor_msgs::PointCloud2 point_output;
    pcl::toROSMsg(*transformed_vehicle_cloud, point_output);
    point_output.header.stamp = cloud_msg->header.stamp;
    point_output.header.frame_id = "Pandar40P";
    pub.publish(point_output);

}

void PlateCloudCallback(const sensor_msgs::PointCloud2ConstPtr& cloud_msg) {
    // 将ROS消息转换为PCL点云格式
    PointCloud::Ptr cloud(new PointCloud);
    PointCloud::Ptr plate_cloud(new PointCloud());
    PlaneCoefficients plate_coefficients;
    pcl::fromROSMsg(*cloud_msg, *cloud);

    // TODO：提取标定版四条边界直线
    ExtractPlateCloud(cloud, plate_cloud, plate_coefficients, 500, 0.03);
    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> lines;
    // ExtractEdgeLines_v2(plate_cloud, lines);

    cloudEdge(plate_cloud, plate_coefficients, lines);

    ros::NodeHandle nh;
    static ros::Publisher plate_pub = nh.advertise<sensor_msgs::PointCloud2>("plate_cloud", 10);
    sensor_msgs::PointCloud2 point_output;
    pcl::toROSMsg(*plate_cloud, point_output);
    point_output.header.stamp = cloud_msg->header.stamp;
    point_output.header.frame_id = "Pandar40P";
    plate_pub.publish(point_output);

    // static ros::Publisher boundary_pub = ros::NodeHandle().advertise<sensor_msgs::PointCloud2>("boundary_cloud", 10);
    // sensor_msgs::PointCloud2 boundary_output;
    // pcl::toROSMsg(*boundary_points, boundary_output);
    // boundary_output.header.stamp = cloud_msg->header.stamp;
    // boundary_output.header.frame_id = "Pandar40P";
    // boundary_pub.publish(boundary_output);

    static ros::Publisher edge_lines_pub = nh.advertise<sensor_msgs::PointCloud2>("edge_lines_cloud", 10);
    for (const auto& edge_line : lines) {
        sensor_msgs::PointCloud2 edge_output;
        pcl::toROSMsg(*edge_line, edge_output);
        edge_output.header.stamp = cloud_msg->header.stamp;
        edge_output.header.frame_id = "Pandar40P";
        edge_lines_pub.publish(edge_output);
    }
}

int main(int argc, char** argv) {
    ros::init(argc, argv, "hesailidar_calib_node");
    ros::NodeHandle nh;
    ros::Subscriber groundcloud_sub = nh.subscribe("/hesai/pandar", 10, GroundCloudCallback);
    // ros::Subscriber plate_sub = nh.subscribe("/hesai/pandar", 10, PlateCloudCallback);
    ros::spin();
    return 0;
}
