#include "PlateExtract.h"

PlateExtractor::PlateExtractor() : plate_plane_(new pcl::PointCloud<pcl::PointXYZI>()){
    plate_coefficients_ = {0.0f, 0.0f, 0.0f, 0.0f};
}

void PlateExtractor::cloudPassThrough(const pcl::PointCloud<pcl::PointXYZI>::Ptr& input_cloud, 
                                      const std::string& axis, float axis_min, float axis_max) {
    pcl::PassThrough<pcl::PointXYZI> passthrough;
    passthrough.setInputCloud(input_cloud);
    passthrough.setFilterFieldName(axis);
    passthrough.setFilterLimits(axis_min, axis_max);
    passthrough.filter(*input_cloud);
}

void PlateExtractor::cloudStatisticalOutlierRemoval(pcl::PointCloud<pcl::PointXYZI>::Ptr cloud) {
    pcl::StatisticalOutlierRemoval<pcl::PointXYZI> sor;   
    sor.setInputCloud (cloud);                            
    // TODO default = 20
    sor.setMeanK (5);                                    // 设置在进行统计时考虑的临近点个数
    sor.setStddevMulThresh (1.0);                         // 设置判断是否为离群点的阀值
    sor.filter (*cloud);
}

void PlateExtractor::Plane_fitting(pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_input,
                                   pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_plane) {
    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
    pcl::SACSegmentation<pcl::PointXYZI> seg;
    // pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_plane(new pcl::PointCloud<pcl::PointXYZ>());

    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setMaxIterations(300);             // 300
    seg.setDistanceThreshold(DisThre);

    // while (cloud_input->size() > 100) {
    //     pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_p(new pcl::PointCloud<pcl::PointXYZI>); // 用来存储从cloud_input中移除当前识别的平面后的剩余点云。
    //     seg.setInputCloud(cloud_input);
    //     seg.segment(*inliers, *coefficients);
    //     if (inliers->indices.size() == 0) {
    //         break;
    //     }
    //     pcl::ExtractIndices<pcl::PointXYZI> extract;
    //     extract.setInputCloud(cloud_input);
    //     extract.setIndices(inliers);
    //     extract.filter(*cloud_plane);     

    //     if (cloud_plane->size() > 100) {
    //         // 输出平面
    //         output_plane(cloud_plane, coefficients, inliers);
    //     }
    //     // 移除plane
    //     extract.setNegative(true);
    //     extract.filter(*cloud_p);
    //     *cloud_input = *cloud_p;
    // }

    pcl::ExtractIndices<pcl::PointXYZI> extract;
    seg.setInputCloud(cloud_input);
    seg.segment(*inliers, *coefficients);
    if (inliers->indices.size() == 0) {
        ROS_WARN("Could not estimate a planar model for the given dataset.");
        return;
    }
    extract.setInputCloud(cloud_input);
    extract.setIndices(inliers);
    extract.setNegative(false);
    extract.filter(*cloud_plane);
    // output_plane(cloud_plane, coefficients, inliers);
    pcl::copyPointCloud(*cloud_plane, *plate_plane_);
    plate_coefficients_.a = coefficients->values[0];
    plate_coefficients_.b = coefficients->values[1];
    plate_coefficients_.c = coefficients->values[2];
    plate_coefficients_.intercept = coefficients->values[3];
    cloudEdge(cloud_plane);
}

void PlateExtractor::output_plane(pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_plane, 
                                  pcl::ModelCoefficients::Ptr coeff, 
                                  pcl::PointIndices::Ptr inliers) {
   // 分割出标定版平面
   if (choicePlane(cloud_plane)) {
        pcl::copyPointCloud(*cloud_plane, *plate_plane_);
        plate_coefficients_.a = coeff->values[0];
        plate_coefficients_.b = coeff->values[1];
        plate_coefficients_.c = coeff->values[2];
        plate_coefficients_.intercept = coeff->values[3];
        // viewer.showCloud(cloud_plane);
        // sleep(SLEEP);
        pcl::io::savePCDFileASCII("myplane.pcd", *cloud_plane);
        std::cout << "save myplane.pcd success !!" << std::endl;

        // 边界提取
        cloudEdge(cloud_plane);
   }
}

// 选取标定板所在平面
bool PlateExtractor::choicePlane(pcl::PointCloud<pcl::PointXYZI>::Ptr cloud) {
    double maxDistance = sqrt(pow(LONG, 2) + pow(WIDE, 2));
    double oldDistance = 0;
    for (std::size_t i = 0; i < cloud->size(); i++) {
        double thisDistance = sqrt(pow(cloud->points[i].x-cloud->points[0].x, 2) 
                                 + pow(cloud->points[i].y-cloud->points[0].y, 2)
                                 + pow(cloud->points[i].z-cloud->points[0].z, 2));
        if (oldDistance < thisDistance) {
            oldDistance = thisDistance;
            if(oldDistance > maxDistance + 0.05) // 0.05
               return false;
        }
    }
    if (oldDistance < LONG)
        return false;
    return true;
}

void PlateExtractor::LineFit3d(pcl::PointCloud<pcl::PointXYZI>::Ptr cloud,
                               float &m, float &n, float &p,
                               pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_line) {
    pcl::SampleConsensusModelLine<pcl::PointXYZI>::Ptr model_line(new pcl::SampleConsensusModelLine<pcl::PointXYZI>(cloud));	
    pcl::RandomSampleConsensus<pcl::PointXYZI> ransac(model_line);	
    ransac.setDistanceThreshold(0.01);	// 内点到模型的最大距离
    ransac.setMaxIterations(1000);		// 最大迭代次数
    ransac.computeModel();				// 执行RANSAC空间直线拟合

    std::vector<int> inliers;				
    ransac.getInliers(inliers);			

    /// 根据索引提取内点
    pcl::copyPointCloud<pcl::PointXYZI>(*cloud, inliers, *cloud_line);

    /// 模型参数
    Eigen::VectorXf coefficient;
    ransac.getModelCoefficients(coefficient);
    m = coefficient[3];
    n = coefficient[4];
    p = coefficient[5];
}

void PlateExtractor::LineFitLeastFit(const std::vector<cv::Point2f> &_points, float &_k, float &_b, float &_r) {
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

double PlateExtractor::CalAngle(double k1, double k2) {
    // 计算两条直线方向向量的点积
    double dotProduct = 1 * 1 + k1 * k2;

    // 计算方向向量的模
    double magnitude1 = std::sqrt(1 * 1 + k1 * k1);
    double magnitude2 = std::sqrt(1 * 1 + k2 * k2);

    double cosTheta = dotProduct / (magnitude1 * magnitude2);

    // 计算夹角
    double theta = std::acos(cosTheta);
    double angleInDegrees = theta * 180.0 / M_PI;
    return angleInDegrees;
}

void PlateExtractor::CalIntersection(cv::Point2f &point, float &A_k, float &A_b, float &B_k, float &B_b) {
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
    coner_points_.push_back(point);
}

void PlateExtractor::SpatialPoint(cv::Point2f &point2D, pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_all) {
    pcl::PointXYZRGB thisColor;
    thisColor.x = -(plate_coefficients_.b * point2D.x + plate_coefficients_.c * point2D.y + plate_coefficients_.intercept) / plate_coefficients_.a;
    thisColor.y = point2D.x;
    thisColor.z = point2D.y;
    thisColor.r = 0;
    thisColor.g = 255;
    thisColor.b = 0;
    cloud_all->push_back(thisColor);
}

void PlateExtractor::SpatialPoint(cv::Point2f &point2D, pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_all, pcl::PointXYZ &thispoint) {
    pcl::PointXYZRGB thisColor;
    thisColor.x = -(plate_coefficients_.b * point2D.x + plate_coefficients_.c * point2D.y + plate_coefficients_.intercept) / plate_coefficients_.a;
    thisColor.y = point2D.x;
    thisColor.z = point2D.y;
    thispoint.x = -(plate_coefficients_.b * point2D.x + plate_coefficients_.c * point2D.y + plate_coefficients_.intercept) / plate_coefficients_.a;
    thispoint.y = point2D.x;
    thispoint.z = point2D.y;
    thisColor.r = 0;
    thisColor.g = 255;
    thisColor.b = 0;
    cloud_all->push_back(thisColor);
}

void PlateExtractor::CalChessPoint(cv::Point2f &pointAb, cv::Point2f &pointBc, cv::Point2f &pointCd, cv::Point2f &pointDa, 
                                   cv::Point2f &innerAb, cv::Point2f &innerBc, cv::Point2f &innerCd, cv::Point2f &innerDa,
                                   pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_all,
                                   pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_chesspoint) {
    // std::vector<cv::Point2f> chess_points;
    chess_points_.clear();

    float margin = 0.2;
    float width = cv::norm(pointBc - pointAb);
    float longth = cv::norm(pointDa - pointAb);
    std::cout << "width = " << width << std::endl;
    std::cout << "longth = " << longth << std::endl;
    float innerWidth = width - 2 * margin;
    float innerLongth = longth - 2 * margin;
    std::cout << "innerwidth = " << innerWidth << std::endl;
    std::cout << "innerlongth = " << innerLongth << std::endl;

    // 计算边长单位向量
    cv::Point2f vectorAB = (pointBc - pointAb) / width;
    cv::Point2f vectorAD = (pointDa - pointAb) / longth;

    // 计算内矩形的四个角点
    innerAb = pointAb + margin * vectorAB + margin * vectorAD;
    innerBc = innerAb + innerWidth * vectorAB;
    innerCd = innerBc + innerLongth * vectorAD;
    innerDa = innerAb + innerLongth * vectorAD;
    chess_points_.push_back(innerAb);
    chess_points_.push_back(innerBc);
    chess_points_.push_back(innerCd);
    chess_points_.push_back(innerDa);

    int rows = 5;
    int cols = 3;
    for (int i = 0; i < cols; i++) {
        for (int j = 0; j < rows; j++) {
            cv::Point2f colStepA = (innerDa - innerAb) * i / (cols - 1);
            cv::Point2f rowStepA = (innerBc - innerAb) * j / (rows - 1);
            cv::Point2f chess_point = innerAb + rowStepA + colStepA;
            chess_points_.push_back(chess_point);
        }
    }
    for (auto i = 0; i < chess_points_.size(); i++) {
        SpatialPoint(chess_points_[i], cloud_all);
        SpatialPoint(chess_points_[i], cloud_chesspoint);
    }
}

void PlateExtractor::addSpatialPoint(cv::Point2f &point2A, cv::Point2f &point2B, 
                     float & _k, float & _b, 
                     pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_all) {
    float step = 0.01;
    if (point2A.x < point2B.x) {
        for (float a = point2A.x; a < point2B.x; a = a + step) {
            cv::Point2f thisPoint;
            thisPoint.x = a;
            thisPoint.y = _k * a + _b;
            SpatialPoint(thisPoint, cloud_all);
        }
    } else {
        for (float a = point2B.x; a < point2A.x; a = a + step) {
            cv::Point2f thisPoint;
            thisPoint.x = a;
            thisPoint.y = _k * a + _b;
            SpatialPoint(thisPoint, cloud_all);
        }
    }
}

void PlateExtractor::bundleAdjustment(const std::vector<Eigen::Vector3d> &pts1, const std::vector<Eigen::Vector3d> &pts2, Eigen::Matrix3d &R, Eigen::Vector3d &t) {
  // 构建图优化，先设定g2o
  typedef g2o::BlockSolverX BlockSolverType;
  typedef g2o::LinearSolverDense<BlockSolverType::PoseMatrixType> LinearSolverType; // 线性求解器类型
  // 梯度下降方法，可以从GN, LM, DogLeg 中选
  auto solver = new g2o::OptimizationAlgorithmLevenberg(std::make_unique<BlockSolverType>(std::make_unique<LinearSolverType>()));
  g2o::SparseOptimizer optimizer;     // 图模型
  optimizer.setAlgorithm(solver);   // 设置求解器
  optimizer.setVerbose(true);       // 打开调试输出

  // vertex
  VertexPose *pose = new VertexPose(); // camera pose
  pose->setId(0);
  pose->setEstimate(Sophus::SE3d());
  optimizer.addVertex(pose);

  // edges
  for (size_t i = 0; i < pts1.size(); i++) {
    EdgeProjectXYZRGBDPoseOnly *edge = new EdgeProjectXYZRGBDPoseOnly(pts2[i]);
    edge->setVertex(0, pose);
    edge->setMeasurement(pts1[i]);
    edge->setInformation(Eigen::Matrix3d::Identity());
    optimizer.addEdge(edge);
  }

  std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
  optimizer.initializeOptimization();
  optimizer.optimize(10);

  // convert to cv::Mat
  R = pose->estimate().rotationMatrix();
  t = pose->estimate().translation();
}

void PlateExtractor::cloudEdge(pcl::PointCloud<pcl::PointXYZI>::Ptr plate_cloud) {
    int index[1000][2];
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
        // cloud_all->push_back(thisColor);
    }

    // 提取边缘
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_edge(new pcl::PointCloud<pcl::PointXYZRGB>);  // 标定版平面边界点云
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_edge_left(new pcl::PointCloud<pcl::PointXYZI>);
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_edge_right(new pcl::PointCloud<pcl::PointXYZI>);
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

    // TODO：输出cloudedge边界线点云（未分离出4条单独直线）
    // viewer.showCloud(cloud_edge);
    // sleep(2);

    

    // 划分4条线，将4条线的点的xy坐标提取出来
    std::vector<cv::Point2f> pointsA;
    std::vector<cv::Point2f> pointsB;
    std::vector<cv::Point2f> pointsC;
    std::vector<cv::Point2f> pointsD;

    // 分离左边边界点云
    double score = 1;
    for (std::size_t i = int(cloud_edge_left->size() / 3); i < int(cloud_edge_left->size() * 2/3) + 1; i ++) {
        pcl::PointCloud<pcl::PointXYZI>::Ptr points_D(new pcl::PointCloud<pcl::PointXYZI>);
        pcl::PointCloud<pcl::PointXYZI>::Ptr points_A(new pcl::PointCloud<pcl::PointXYZI>);
        for (std::size_t j = 0; j < i; j ++) {
            points_D->push_back(cloud_edge_left->points[j]);
        }
        for (std::size_t j = i; j < cloud_edge_left->size(); j ++) {
            points_A->push_back(cloud_edge_left->points[j]);
        }
        float D_m, D_n, D_p;
        float A_m, A_n, A_p;
        pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_line_D(new pcl::PointCloud<pcl::PointXYZI>);
        pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_line_A(new pcl::PointCloud<pcl::PointXYZI>);
        // 
        LineFit3d(points_D, D_m, D_n, D_p, cloud_line_D);
        LineFit3d(points_A, A_m, A_n, A_p, cloud_line_A);
        Eigen::Vector3d v_D(D_m, D_n, D_p);
        Eigen::Vector3d v_A(A_m, A_n, A_p);

        double s = abs(v_D.dot(v_A) / (v_D.norm() * v_A.norm()));

        if(s < score) {
            for(std::size_t k = 0; k < cloud_line_D->size(); k++) {
                cv::Point2f thisPoint;
                thisPoint.x = cloud_line_D->points[k].y;
                thisPoint.y = cloud_line_D->points[k].z;
                pointsD.push_back(thisPoint);
            }
            for(std::size_t k = 0; k < cloud_line_A->size(); k++) {
                cv::Point2f thisPoint;
                thisPoint.x = cloud_line_A->points[k].y;
                thisPoint.y = cloud_line_A->points[k].z;
                pointsA.push_back(thisPoint);
            }
            score = s;
        }
    }
    std::cout << "sort line A & line D points successfully !!!" << std::endl;

    // 分离右边边界点云
    score = 1;
    for(std::size_t i = int(cloud_edge_right->size() / 3); i < int(cloud_edge_right->size() * 2/3) + 1; i ++) {
        pcl::PointCloud<pcl::PointXYZI>::Ptr points_B(new pcl::PointCloud<pcl::PointXYZI>);
        pcl::PointCloud<pcl::PointXYZI>::Ptr points_C(new pcl::PointCloud<pcl::PointXYZI>);
        for(std::size_t j = 0; j < i; j ++){
            points_B->push_back(cloud_edge_right->points[j]);
        }
        for(std::size_t j = i; j < cloud_edge_right->size(); j++){
            points_C->push_back(cloud_edge_right->points[j]);
        }
        float B_m, B_n, B_p;
        float C_m, C_n, C_p;
        pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_line_B(new pcl::PointCloud<pcl::PointXYZI>);
        pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_line_C(new pcl::PointCloud<pcl::PointXYZI>);

        LineFit3d(points_B, B_m, B_n, B_p, cloud_line_B);
        LineFit3d(points_C, C_m, C_n, C_p, cloud_line_C);
        Eigen::Vector3d v_B(B_m, B_n, B_p);
        Eigen::Vector3d v_C(C_m, C_n, C_p);
        double s = abs(v_B.dot(v_C) / (v_B.norm() * v_C.norm()));
        // std::cout << s << std::endl;
        if(s < score){
            for(std::size_t k = 0; k < cloud_line_B->size(); k ++){
                cv::Point2f thisPoint;
                thisPoint.x = cloud_line_B->points[k].y;
                thisPoint.y = cloud_line_B->points[k].z;
                pointsB.push_back(thisPoint);
            }
            for(std::size_t k = 0; k < cloud_line_C->size(); k ++){
                cv::Point2f thisPoint;
                thisPoint.x = cloud_line_C->points[k].y;
                thisPoint.y = cloud_line_C->points[k].z;
                pointsC.push_back(thisPoint);
            }
            score = s;
        }
    }
    std::cout << "sort line B & line C points successfully !!!" << std::endl;
    
    float A_k, A_b, A_r;
    float B_k, B_b, B_r;
    float C_k, C_b, C_r;
    float D_k, D_b, D_r;

    // 二维直线拟合
    LineFitLeastFit(pointsA, A_k, A_b, A_r);
    LineFitLeastFit(pointsB, B_k, B_b, B_r);
    LineFitLeastFit(pointsC, C_k, C_b, C_r);
    LineFitLeastFit(pointsD, D_k, D_b, D_r);

    // 验证两条平面直线的夹角是否为90度
    double angle_AB = CalAngle(A_k, B_k);
    double angle_AD = CalAngle(A_k, D_k);
    double angle_BC = CalAngle(B_k, C_k);
    double angle_CD = CalAngle(C_k, D_k);
    std::cout << "angle_AB = " << angle_AB << std::endl;
    std::cout << "angle_AD = " << angle_AD << std::endl;
    std::cout << "angle_BC = " << angle_BC << std::endl;
    std::cout << "angle_CD = " << angle_CD << std::endl;

    // 求4条直线的交点
    cv::Point2f pointAb, pointBc, pointCd, pointDa;  // 4个相交点的二维坐标
    coner_points_.clear();
    CalIntersection(pointAb, A_k, A_b, B_k, B_b);
    CalIntersection(pointBc, B_k, B_b, C_k, C_b);
    CalIntersection(pointCd, C_k, C_b, D_k, D_b);
    CalIntersection(pointDa, D_k, D_b, A_k, A_b);

    // 求棋盘格角点
    cv::Point2f point_a, point_b, point_c, point_d;  // 4个棋盘格角点的二维坐标
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_chesspoint(new pcl::PointCloud<pcl::PointXYZRGB>);  // 标定版平面边界点云
    CalChessPoint(pointAb, pointBc, pointCd, pointDa, 
                  point_a, point_b, point_c, point_d,
                  cloud_all, cloud_chesspoint);

    // 求边界直线4个交点的空间坐标 
    SpatialPoint(pointAb, cloud_all);
    SpatialPoint(pointBc, cloud_all);
    SpatialPoint(pointCd, cloud_all);
    SpatialPoint(pointDa, cloud_all);

    // 增加空间点
    addSpatialPoint(pointAb, pointBc, B_k, B_b, cloud_all);
    addSpatialPoint(pointBc, pointCd, C_k, C_b, cloud_all);
    addSpatialPoint(pointCd, pointDa, D_k, D_b, cloud_all);
    addSpatialPoint(pointDa, pointAb, A_k, A_b, cloud_all);

    // 保存边界点云
    // viewer.showCloud(cloud_edge);
    // sleep(SLEEP);
    pcl::io::savePCDFileASCII("myEdge.pcd", *cloud_edge);
    cout << "save myEdge.pcd success !!" << endl;

    // 保存拟合的边界点云 + 棋盘格角点点云
    // viewer.showCloud(cloud_all);
    // sleep(SLEEP);
    pcl::io::savePCDFileASCII("myEstimate.pcd", *cloud_all);
    cout << "save myEstimate.pcd success !!" << endl;

    // 保存棋盘格点云
    pcl::io::savePCDFileASCII("myChessboard.pcd", *cloud_chesspoint);
    cout << "save myChessboard.pcd success !!" << endl;
}