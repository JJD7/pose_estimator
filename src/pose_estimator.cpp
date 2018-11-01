#include "pose_estimator.h"


pose_estimator::pose_estimator(ros::NodeHandle* nodehandle):nh_(*nodehandle)
{ // constructor
    ROS_INFO("in class constructor of ExampleRosClass");
}

void imageCallback(const sensor_msgs::ImageConstPtr& msg)
{
  try
  {
    cv::imshow("view", cv_bridge::toCvShare(msg, "bgr8")->image);
    cv::waitKey(30);
  }
  catch (cv_bridge::Exception& e)
  {
    ROS_ERROR("Could not convert from '%s' to 'bgr8'.", msg->encoding.c_str());
  }
}

int main(int argc, char **argv){
    // required to access ros. If this gives an error, make sure you are running
    // roscore on your machine.
    ros::init(argc,argv,"pose_estimator");
    ros::NodeHandle nh;
    ros::Duration(.1).sleep();


    cv::namedWindow("left_rectified_image");
    cv::startWindowThread();
    ROS_INFO("Initializing Subscribers");
    image_transport::ImageTransport it(nh);
    image_transport::Subscriber leftRect_image= it.subscribe("/left_right/left_rect/image_raw", 1, imageCallback);


    ros::Publisher pose_estimate;
    ROS_INFO("Initializing Publishers");
    pose_estimate = nh.advertise<geometry_msgs::Pose>("estimated_pose",1);

    ROS_INFO("Running");

    while(ros::ok()){

            ros::spinOnce();

    }

    cv::destroyWindow("left_rectified_image");
    return 0;
}
