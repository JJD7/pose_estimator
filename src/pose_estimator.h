#ifndef POSE_ESTIMATOR_H
#define POSE_ESTIMATOR_H

#include <iostream>
#include <ros/ros.h>
#include <ros/console.h>
#include <tf/transform_listener.h>
#include <tf/tf.h>
#include <geometry_msgs/Twist.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/Pose.h>
#include <nav_msgs/Odometry.h>
#include <std_msgs/Float32.h>
#include <math.h>
#include <string>
#include <image_transport/image_transport.h>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>

class pose_estimator
{
    public:
    pose_estimator(ros::NodeHandle* nodehandle);
    void imgCallback(const  sensor_msgs::ImageConstPtr& msg);

    private:
    ros::NodeHandle nh_;
    geometry_msgs::Pose est_pose;

};

#endif // POSE_ESTIMATOR_H
