#ifndef _GROUNDEXTRACT_H
#define _GROUNDEXTRACT_H

#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <iostream>
#include <math.h>
#include <cmath>
#include <string.h>
//常用点云类
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/filters/passthrough.h>  //直通滤波器头文件
#include <pcl/filters/statistical_outlier_removal.h> //滤波相关
#include <pcl/visualization/cloud_viewer.h>   //类cloud_viewer头文件申明
//分割类
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/sample_consensus/sac_model_line.h>
#include <pcl/sample_consensus/ransac.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/filters/extract_indices.h>
// 图像类
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <Eigen/Eigen>
#include <Eigen/Core>
// 稠密矩阵的代数运算（逆，特征值等）
#include <Eigen/Dense>

// using namespace pcl;
// using namespace cv;
// using namespace std;

#define Axis2Front 3.625     // 数模数据
#define Axis2Rear 1.026
#define Axis2DoorX 1.128
#define Axis2DoorY 0.91

class GroundExtractor {
public:
    struct PlaneCoefficients {
        float a;
        float b;
        float c;
        float intercept;
    };

    GroundExtractor();
    // 直通滤波
    void cloudPassThrough(const pcl::PointCloud<pcl::PointXYZ>::Ptr& input_cloud, 
                        const std::string& axis, float axis_min, float axis_max);

    // 根据R T求lidar相对于地面的变换矩阵
    Eigen::Matrix4d GetMatrix(const Eigen::Vector3d& translation, const Eigen::Matrix3d& rotation);

    // 由数模数据求yaw角
    double GetYaw(double front_x, double front_y, double rear_x, double rear_y);
    
    // 从旋转矩阵R中求roll角
    double GetRoll(const Eigen::Matrix4d& matrix);

    // 从旋转矩阵R中求pitch角
    double GetPitch(const Eigen::Matrix4d& matrix);

    Eigen::Vector3d GetXYZ(Eigen::Matrix3d rotation);

    // 提取地面点云
    bool ExtractGroundCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, 
                            pcl::PointCloud<pcl::PointXYZ>::Ptr& ground_cloud, 
                            int max_iterations, float distance_threshold);

    const PlaneCoefficients& getGroundCoefficients() const { return ground_coefficients_; }

private:
    PlaneCoefficients ground_coefficients_; 
};


#endif // _GROUNDEXTRACT_H