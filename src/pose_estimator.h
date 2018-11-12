#ifndef POSE_ESTIMATOR_H
#define POSE_ESTIMATOR_H

//Standard includes
#include <iostream>
#include <fstream>
#include <string>
#include <math.h>
#include <stdio.h>
#include <Eigen/Geometry>

//Ros includes
#include <ros/ros.h>
#include <ros/console.h>

#include <tf/transform_listener.h>
#include <tf/transform_datatypes.h>
#include <tf/tf.h>

#include <geometry_msgs/Twist.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/PoseWithCovariance.h>
#include <nav_msgs/Odometry.h>

#include <std_msgs/Float32.h>
#include <image_transport/image_transport.h>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <sensor_msgs/Image.h>
#include <stereo_msgs/DisparityImage.h>



//OpenCV Includes
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/opencv_modules.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/stitching/detail/matchers.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
//#include <opencv2/flann.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>

//PCL Includes
#include <pcl/point_cloud.h>
#include <pcl/common/transforms.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>
//#include <pcl/registration/transformation_estimation.h>
//#include <pcl/registration/transformation_estimation_svd.h>
//#include <pcl/registration/transformation_estimation_svd_scale.h>
//#include <pcl/filters/statistical_outlier_removal.h>


using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;
using namespace cv::detail;
using namespace sensor_msgs;
using namespace message_filters;

class pose_estimator
{
    public:
    pose_estimator(ros::NodeHandle* nodehandle);
    ros::Publisher pose_estimate;

    double focallength = 16.0 / 1000 / 3.75 * 1000000;
    double baseline = 600.0 / 1000;
    double minDisparity = 64;
    int rows = 0, cols = 0, cols_start_aft_cutout = 0;
    int blur_kernel = 1;
    int boundingBox = 20;
    int cutout_ratio = 8; //ratio of masking to be done on left side of image as this area is not covered in stereo disparity images.
    string calib_file = "cam13calib.yml";
    string calib_file_Dir = "/home/jd/catkin_ws/src/ros_multi_baseline_stereo/ros_sgm/config/";
    Mat Q;
    const int featureMatchingThreshold = 100;
    float dx_FM, dy_FM, dz_FM, droll_FM, dpitch_FM, dyaw_FM;

    ////translation and rotation between image and head of hexacopter
    //const double trans_x_hi = -0.300;
    //const double trans_y_hi = -0.040;
    //const double trans_z_hi = -0.350;
    //const double PI = 3.141592653589793238463;
    //const double theta_xi = -1.1408 * PI / 180;
    //const double theta_yi = 1.1945 * PI / 180;

    // Variables to store keypoints and descriptors
    Ptr<FeaturesFinder> finder;
    pcl::IterativeClosestPoint<pcl::PointXYZRGB, pcl::PointXYZRGB> icp;
    ImageFeatures features1, features2;	//has features.keypoints and features.descriptors
    geometry_msgs::Pose est_pose, pose_ekf1, pose_ekf2;
    std::vector<DMatch> matches;
    std::vector<Point2f> points1, points2;
    Mat im1Gray, im2Gray, im1RGB, im2RGB;
    Mat disparity1, disparity2;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr keypoints3D1, keypoints3D2;
    vector<bool> keypoints3D_ROI_Points1, keypoints3D_ROI_Points2;

    //Function Declarations
    void createImgPtCloud(Mat &im, pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloudrgb);
    void generate_FM_Transform();
    void extract_features();
    void init(Mat &im);
    void estimatePose();
    double getMean(Mat disp_img);
    double getVariance(Mat disp_img);
    void generatePose();


    private:
    ros::NodeHandle nh_;
    void readCalibFile();
};

#endif // POSE_ESTIMATOR_H
