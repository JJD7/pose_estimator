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
using namespace cv::detail;

class pose_estimator
{
    public:
    pose_estimator(ros::NodeHandle* nodehandle);
    bool visualize = false;
    bool align_point_cloud = false;
    string read_PLY_filename0 = "";
    string read_PLY_filename1 = "";
    double focallength = 16.0 / 1000 / 3.75 * 1000000;
    double baseline = 600.0 / 1000;
    int cutout_ratio = 8; //ratio of masking to be done on left side of image as this area is not covered in stereo disparity images.
    string calib_file = "cam13calib.yml";
    string calib_file_Dir = "/home/jd/catkin_ws/src/ros_multi_baseline_stereo/ros_sgm/config/";
    Mat Q;
	
    private:
    ros::NodeHandle nh_;
    geometry_msgs::Pose est_pose;
    void readCalibFile();

};

class RawImageData {
public:
	int img_num;
	Mat rgb_image;
	Mat disparity_image;
	Mat segment_label;
	Mat double_disparity_image;
	
	double time;	//NSECS
	double tx;
	double ty;
	double tz;
	double qx;
	double qy;
	double qz;
	double qw;
};

class ImageData {
public:
	RawImageData* raw_img_data_ptr;
	
	ImageFeatures features;	//has features.keypoints and features.descriptors
	cuda::GpuMat gpu_descriptors;
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr keypoints3D;
	vector<bool> keypoints3D_ROI_Points;
	
	pcl::registration::TransformationEstimation<pcl::PointXYZRGB, pcl::PointXYZRGB>::Matrix4 t_mat_MAVLink;
	pcl::registration::TransformationEstimation<pcl::PointXYZRGB, pcl::PointXYZRGB>::Matrix4 t_mat_FeatureMatched;
	
};

#endif // POSE_ESTIMATOR_H
