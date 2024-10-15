#ifndef _PLATERXTRACT_H
#define _PLATERXTRACT_H

#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <iostream>
#include <math.h>
#include <cmath>
#include <string.h>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/filters/passthrough.h>  //直通滤波器头文件
#include <pcl/filters/statistical_outlier_removal.h> //滤波相关
#include <pcl/visualization/cloud_viewer.h>   //类cloud_viewer头文件申明

#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/sample_consensus/sac_model_line.h>
#include <pcl/sample_consensus/ransac.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/filters/extract_indices.h>
#include <pcl_ros/point_cloud.h>

#include <Eigen/Eigen>
#include <Eigen/Core>
#include <Eigen/Dense>

#include <g2o/core/base_vertex.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <sophus/se3.hpp>

// pcl::visualization::CloudViewer viewer("viewer");

#define LONG 0.76      // 标定板长、宽
#define WIDE 0.58
#define DisThre 0.03   // 平面分割阈值
#define SLEEP 2        // 睡眠时间

class PlateExtractor {
public:
    struct PlaneCoefficients {
        float a;
        float b;
        float c;
        float intercept;
    }; 

    PlateExtractor();

    // 直通滤波
    void cloudPassThrough(const pcl::PointCloud<pcl::PointXYZI>::Ptr& input_cloud, 
                          const std::string& axis, float axis_min, float axis_max);

    // 去除离群点
    void cloudStatisticalOutlierRemoval(pcl::PointCloud<pcl::PointXYZI>::Ptr cloud);

    // 平面提取
    void Plane_fitting(pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_input,
                       pcl::PointCloud<pcl::PointXYZI>::Ptr plate_cloud);

    // 输出平面
    void output_plane(pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_plane, 
                      pcl::ModelCoefficients::Ptr coeff, pcl::PointIndices::Ptr inliers);

    // 筛选出标定版平面
    bool choicePlane(pcl::PointCloud<pcl::PointXYZI>::Ptr cloud);

    // 从边界点云中分离出单独的2条直线
    void LineFit3d(pcl::PointCloud<pcl::PointXYZI>::Ptr cloud, float &m, float &n, float &p, pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_line);

    // 二维直线拟合
    void LineFitLeastFit(const std::vector<cv::Point2f> &_points, float &_k, float &_b, float &_r);

    // 计算两条二维直线的夹角
    double CalAngle(double k1, double k2);

    // 计算两条二维平面直线的交点
    void CalIntersection(cv::Point2f &point, float &A_k, float &A_b, float &B_k, float &B_b);

    // 求二维点在空间中的坐标
    void SpatialPoint(cv::Point2f &point2D, pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_all);

    void SpatialPoint(cv::Point2f &point2D, pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_all, pcl::PointXYZ &thispoint);

    // 求棋盘格角点
    void CalChessPoint(cv::Point2f &pointAb, cv::Point2f &pointBc, cv::Point2f &pointCd, cv::Point2f &pointDa, 
                       cv::Point2f &innerAb, cv::Point2f &innerBc, cv::Point2f &innerCd, cv::Point2f &innerDa,
                       pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_all,
                       pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_chesspoint);

    // 添加空间直线上的点 
    void addSpatialPoint(cv::Point2f &point2A, cv::Point2f &point2B, float &_k, float &_b, 
                         pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_all);

    // ICP处理
    void bundleAdjustment(const std::vector<Eigen::Vector3d> &pts1, 
                          const std::vector<Eigen::Vector3d> &pts2,
                          Eigen::Matrix3d &R, 
                          Eigen::Vector3d &t);

    // 提取标定版平面边界直线
    void cloudEdge(pcl::PointCloud<pcl::PointXYZI>::Ptr plate_cloud);
    
    const PlaneCoefficients& getPlateCoefficients() const { return plate_coefficients_; }
    const std::vector<cv::Point2f>& getConerPoints() const { return coner_points_; }
    const std::vector<cv::Point2f>& getChessPoints() const { return chess_points_; }
    pcl::PointCloud<pcl::PointXYZI>::Ptr getPlatePlane() const { return plate_plane_; }
    
private:
    PlaneCoefficients plate_coefficients_;
    std::vector<cv::Point2f> coner_points_;
    std::vector<cv::Point2f> chess_points_;
    pcl::PointCloud<pcl::PointXYZI>::Ptr plate_plane_;
};

// vertex and edges used in g2o ba
class VertexPose : public g2o::BaseVertex<6, Sophus::SE3d> {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
  virtual void setToOriginImpl() override {
    _estimate = Sophus::SE3d();
  }

  /// left multiplication on SE3
  virtual void oplusImpl(const double *update) override {
    Eigen::Matrix<double, 6, 1> update_eigen;
    update_eigen << update[0], update[1], update[2], update[3], update[4], update[5];
    _estimate = Sophus::SE3d::exp(update_eigen) * _estimate;
  }

  virtual bool read(istream &in) override {}
  virtual bool write(ostream &out) const override {}
};

// g2o edge
class EdgeProjectXYZRGBDPoseOnly : public g2o::BaseUnaryEdge<3, Eigen::Vector3d, VertexPose> {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

  EdgeProjectXYZRGBDPoseOnly(const Eigen::Vector3d &point) : _point(point) {}

  virtual void computeError() override {
    const VertexPose *pose = static_cast<const VertexPose *> ( _vertices[0] );
    _error = _measurement - pose->estimate() * _point;
  }

  virtual void linearizeOplus() override {
    VertexPose *pose = static_cast<VertexPose *>(_vertices[0]);
    Sophus::SE3d T = pose->estimate();
    Eigen::Vector3d xyz_trans = T * _point;
    _jacobianOplusXi.block<3, 3>(0, 0) = -Eigen::Matrix3d::Identity();
    _jacobianOplusXi.block<3, 3>(0, 3) = Sophus::SO3d::hat(xyz_trans);
  }

  bool read(istream &in) {}
  bool write(ostream &out) const {}

protected:
  Eigen::Vector3d _point;
};

#endif // _PLATERXTRACT_H