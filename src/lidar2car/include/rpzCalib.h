#ifndef RPZ_CALIB_RPZ_CALIB_H_
#define RPZ_CALIB_RPZ_CALIB_H_

#include <iostream>
#include <string>
#include <vector>
#include <dirent.h>
#include <chrono>

#include "GroundExtractor.h"

class RPCalib{
    public:
        // RPCalib(std::string output_dir);
        RPCalib();
        ~RPCalib();

        bool LoadData(std::string dataset_folder, const std::vector<Eigen::Matrix4d> &lidar_pose, int start_frame, int end_frame);
        bool Calibrate_old(Eigen::Matrix4d & extrinsic);
        PointCloudPtr Calibrate(pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud);
        bool GroundPlaneExtraction(const PointCloudPtr &in_cloud, PointCloudPtr g_cloud, PointCloudPtr ng_cloud, PlaneParam &plane);

    private:
        std::vector<double> trans_;
        std::vector<double> rollset_;
        std::vector<double> pitchset_;

        std::string lidar_path_;
        std::string output_dir_;
        std::vector<std::string> lidar_files_;
        std::unique_ptr<GroundExtractor> ground_extractor_;
        std::vector<double> lidar_pose_y_;
        int file_num_;

        // param
        const double master_normal_check_ = 0.8;
        const double FILTER_MAX_RANGE_ = 150;
        const double FILTER_MIN_RANGE_ = 1;
        const int DOWN_SAMPLE_ = 5;
};

#endif  //  RPZ_CALIB_RPZ_CALIB_H_