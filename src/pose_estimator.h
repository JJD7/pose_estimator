#ifndef POSE_ESTIMATOR_H
#define POSE_ESTIMATOR_H

//Standard includes
#include <iostream>
#include <fstream>
#include <string>
#include <math.h>
#include <stdio.h>

//Ros includes
#include <ros/ros.h>
#include <ros/console.h>
#include <tf/transform_listener.h>
#include <tf/tf.h>
#include <geometry_msgs/Twist.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/Pose.h>
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
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/cudafeatures2d.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/flann.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>

//PCL Includes
#include <pcl/io/ply_io.h>
#include <pcl/point_cloud.h>
#include <pcl/console/parse.h>
#include <pcl/common/transforms.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/point_types.h>
#include <pcl/registration/transformation_estimation.h>
#include <pcl/registration/transformation_estimation_svd.h>
#include <pcl/registration/transformation_estimation_svd_scale.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/features/normal_3d.h>
#include <pcl/registration/icp.h>
#include <pcl/surface/mls.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/surface/gp3.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/filters/project_inliers.h>
#include <pcl/surface/concave_hull.h>

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
    bool visualize = false;
    bool align_point_cloud = false;
    string read_PLY_filename0 = "";
    string read_PLY_filename1 = "";
    double focallength = 16.0 / 1000 / 3.75 * 1000000;
    double baseline = 600.0 / 1000;
    double minDisparity = 64;
    int rows = 0, cols = 0, cols_start_aft_cutout = 0;
    int blur_kernel = 1;
    int jump_pixels = 10;
    int boundingBox = 20;
    int cutout_ratio = 8; //ratio of masking to be done on left side of image as this area is not covered in stereo disparity images.
    string calib_file = "cam13calib.yml";
    string calib_file_Dir = "/home/jd/catkin_ws/src/ros_multi_baseline_stereo/ros_sgm/config/";
    Mat Q;
    const int featureMatchingThreshold = 100;

    double tx;
    double ty;
    double tz;
    double qx;
    double qy;
    double qz;
    double qw;

    //translation and rotation between image and head of hexacopter
    const double trans_x_hi = -0.300;
    const double trans_y_hi = -0.040;
    const double trans_z_hi = -0.350;
    const double PI = 3.141592653589793238463;
    const double theta_xi = -1.1408 * PI / 180;
    const double theta_yi = 1.1945 * PI / 180;

    // Variables to store keypoints and descriptors
    Ptr<FeaturesFinder> finder;
    //Ptr<cv::DescriptorMatcher> matcher = cv::BFMatcher(cv::NORM_HAMMING);
    //Ptr<cuda::DescriptorMatcher> matcher = cv::cuda::DescriptorMatcher::createBFMatcher(cv::NORM_HAMMING);
    ImageFeatures features1, features2;	//has features.keypoints and features.descriptors
    cuda::GpuMat gpu_descriptors1, gpu_descriptors2;
    //std::vector<KeyPoint> keypoints1, keypoints2;
    //Mat descriptors1, descriptors2;
    geometry_msgs::Pose est_pose, pose_ekf1, pose_ekf2;
    std::vector<DMatch> matches;
    std::vector<Point2f> points1, points2;
    Mat im1Gray, im2Gray, im1RGB, im2RGB;
    Mat disparity1, disparity2;
    Mat orb_img1, orb_img2;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr keypoints3D1, keypoints3D2;
    vector<bool> keypoints3D_ROI_Points1, keypoints3D_ROI_Points2; 
    int generate_Matched_Keypoints_Point_Cloud(pcl::PointCloud<pcl::PointXYZRGB>::Ptr &current_img_matched_keypoints, pcl::PointCloud<pcl::PointXYZRGB>::Ptr &fitted_cloud_matched_keypoints);

    //Function Declarations
    void createImgPtCloud(Mat &im, pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloudrgb);
    void extract_features(Mat &im);
    void init(Mat &im);
    void estimatePose();
    double getMean(Mat disp_img);
    double getVariance(Mat disp_img);

    pcl::registration::TransformationEstimation<pcl::PointXYZRGB, pcl::PointXYZRGB>::Matrix4 t_mat_MAVLink1, t_mat_MAVLink2;
    pcl::registration::TransformationEstimation<pcl::PointXYZRGB, pcl::PointXYZRGB>::Matrix4 t_mat_FeatureMatched1, t_mat_FeatureMatched2;
    void transformPtCloud(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloudrgb, pcl::PointCloud<pcl::PointXYZRGB>::Ptr transformed_cloudrgb, pcl::registration::TransformationEstimation<pcl::PointXYZRGB, pcl::PointXYZRGB>::Matrix4 transform);
    pcl::registration::TransformationEstimation<pcl::PointXYZRGB, pcl::PointXYZRGB>::Matrix4 generateTmat(geometry_msgs::Pose& pose);
    pcl::PointXYZRGB generateUAVpos();
    pcl::registration::TransformationEstimation<pcl::PointXYZRGB, pcl::PointXYZRGB>::Matrix4 generate_tf_of_Matched_Keypoints(bool &acceptDecision);
    pcl::PointXYZRGB transformPoint(pcl::PointXYZRGB hexPosMAVLink, pcl::registration::TransformationEstimation<pcl::PointXYZRGB, pcl::PointXYZRGB>::Matrix4 T_SVD_matched_pts);

    private:
    ros::NodeHandle nh_;
    void readCalibFile();
};

#endif // POSE_ESTIMATOR_H
