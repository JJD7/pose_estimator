#include "pose_estimator.h"

const int MAX_FEATURES = 1000;
const float GOOD_MATCH_PERCENT = 0.15f;
bool initializing = true;
double accepted_im_count =0;
bool acceptDecision = true;

PoseEstimate::PoseEstimate(ros::NodeHandle* nodehandle):nh_(*nodehandle)
{

    calib_file = "cam13calib.yml";
    calib_file_Dir = "/home/jd/catkin_ws/src/ros_multi_baseline_stereo/ros_sgm/config/";

    ROS_INFO("in class constructor of PoseEstimate");
    readCalibFile();


    focallength = 16.0 / 1000 / 3.75 * 1000000;
    baseline = 600.0 / 1000;
    minDisparity = 64;
    rows = 0;
    cols = 0;
    cols_start_aft_cutout = 0;
    blur_kernel = 1;
    boundingBox = 20;
    cutout_ratio = 8;
    _estimator_weight = 0;
    featureMatchingThreshold = 100;


    ROS_INFO("Initializing Publishers");
    pose_estimate = nh_.advertise<geometry_msgs::PoseStamped>("estimated_pose",1);

    dynamic_reconfigure::Server<pose_estimator::EstimatorConfig>::CallbackType f;
    f = boost::bind(&PoseEstimate::estimator_reconfig_cb, this, _1, _2);
    _estimator_cfg_server.setCallback(f);

}

void PoseEstimate::readCalibFile()
{
        ROS_INFO("Loading Calibrati on File");
        cv::FileStorage fs(calib_file_Dir + calib_file, cv::FileStorage::READ);
        fs["Q"] >> Q;
        if(Q.empty())
                throw "Exception: could not read Q matrix";
        fs.release();
        ROS_INFO("Loaded Camera Q matrix as: ");
        std::cout << Q << endl;
}

void PoseEstimate::extract_features()
{
    // Convert images to grayscale
    cvtColor(im2RGB, im2Gray, CV_BGR2GRAY);

    finder = makePtr<OrbFeaturesFinder>();
    (*finder)(im2RGB, features2);

    //cuda::GpuMat descriptor(features2.descriptors);
    //gpu_descriptors2 = descriptor;

    vector<KeyPoint> keypoints = features2.keypoints;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr keypoints3dptcloud (new pcl::PointCloud<pcl::PointXYZRGB> ());
    keypoints3dptcloud->is_dense = true;

    vector<bool> pointsInROIVec;

    int good = 0, bad = 0;
    for (int i = 0; i < keypoints.size(); i++)
    {
            double disp_value;
            disp_value = (double)disparity2.at<char>(keypoints[i].pt.y, keypoints[i].pt.x);
            //cout << "disparity val = " << disp_value << endl;
            cv::Mat_<double> vec_src(4, 1);

            double xs = keypoints[i].pt.x;
            double ys = keypoints[i].pt.y;

            vec_src(0) = xs; vec_src(1) = ys; vec_src(2) = disp_value; vec_src(3) = 1;
            vec_src = Q * vec_src;
            vec_src /= vec_src(3);

            pcl::PointXYZRGB pt_3d_src;

            pt_3d_src.x = vec_src(0);
            pt_3d_src.y = vec_src(1);
            pt_3d_src.z = vec_src(2);

            keypoints3dptcloud->points.push_back(pt_3d_src);

            if (disp_value > minDisparity && keypoints[i].pt.x >= cols_start_aft_cutout)
            {
                    good++;
                    pointsInROIVec.push_back(true);
            }
            else
            {
                    bad++;
                    pointsInROIVec.push_back(false);
            }
    }

    //cout << " g" << good << "/b" << bad << flush <<endl;
    keypoints3D2 = keypoints3dptcloud;
    keypoints3D_ROI_Points2 = pointsInROIVec;
}

void PoseEstimate::generate_FM_Transform()
{
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr keypoints3D_src = keypoints3D2;
    vector<bool> pointsInROIVec_src = keypoints3D_ROI_Points2;

    BFMatcher matcher(NORM_HAMMING);
    vector<vector<DMatch> > matches;
    matcher.knnMatch(features2.descriptors, features1.descriptors, matches, 2);
    //matcher->knnMatch(gpu_descriptors2, gpu_descriptors1, matches, 2);

    vector<DMatch> good_matches;
    for(int k = 0; k < matches.size(); k++)
    {
        if(matches[k][0].distance < 0.5 * matches[k][1].distance && matches[k][0].distance < 40)
        {
            good_matches.push_back(matches[k][0]);
        }
    }

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr keypoints3D_dst = keypoints3D1;

    vector<bool> pointsInROIVec_dst = keypoints3D_ROI_Points1;

    //using sequential matched points to estimate the rigid body transformation between matched 3D points
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_current_temp (new pcl::PointCloud<pcl::PointXYZRGB> ());
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_prior_temp (new pcl::PointCloud<pcl::PointXYZRGB> ());
    cloud_current_temp->is_dense = true;
    cloud_prior_temp->is_dense = true;

    for (int match_index = 0; match_index < good_matches.size(); match_index++)
    {
        DMatch match = good_matches[match_index];

        int dst_Idx = match.trainIdx;	//dst img
        int src_Idx = match.queryIdx;	//src img

        if(pointsInROIVec_src[src_Idx] == true && pointsInROIVec_dst[dst_Idx] == true)
        {
            cloud_current_temp->points.push_back(keypoints3D_src->points[src_Idx]);
            cloud_prior_temp->points.push_back(keypoints3D_dst->points[dst_Idx]);
        }
    }

    if (good_matches.size() < featureMatchingThreshold) //dont estimate transform from this set
    {
        acceptDecision = false;
    }
    else //good enough to estimate transorm
    {
        icp.setInputSource(cloud_prior_temp);
        icp.setInputTarget(cloud_current_temp);
        pcl::PointCloud<pcl::PointXYZRGB> Final;
        icp.align(Final);
        std::cout << "has converged:" << icp.hasConverged() << " score: " <<
        icp.getFitnessScore() << std::endl;

        // source: https://stackoverflow.com/questions/18956151/icp-transformation-matrix-interpretation
        Eigen::Matrix4f trafo = icp.getFinalTransformation();
        Eigen::Transform<float, 3, Eigen::Affine> tROTA(trafo);

        pcl::getTranslationAndEulerAngles(tROTA, dx_FM, dy_FM, dz_FM, droll_FM, dpitch_FM, dyaw_FM);
    }

}

void PoseEstimate::init(Mat &im)
{
    ROS_INFO("initalizing tracker");
    rows = im.rows;
    cols = im.cols;
    cols_start_aft_cutout = (int)(cols/cutout_ratio);

    getTransform();
    initializing = false;
}

double PoseEstimate::getMean(Mat disp_img)
{
    double sum = 0.0;
    for (int y = boundingBox; y < rows - boundingBox; ++y)
    {
        for (int x = cols_start_aft_cutout; x < cols - boundingBox; ++x)
        {
            double disp_val = 0;
            disp_val = (double)disp_img.at<char>(y,x);

            if (disp_val > minDisparity)
                sum += disp_val;
        }
    }
    return sum/((rows - 2 * boundingBox )*(cols - boundingBox - cols_start_aft_cutout));
}

double PoseEstimate::getVariance(Mat disp_img)
{
    double mean = getMean(disp_img);
    double temp = 0;

    for (int y = boundingBox; y < rows - boundingBox; ++y)
    {
        for (int x = cols_start_aft_cutout; x < cols - boundingBox; ++x)
        {
            double disp_val = 0;
            disp_val = (double)disp_img.at<char>(y,x);

            if (disp_val > minDisparity)
                temp += (disp_val-mean)*(disp_val-mean);
        }
    }
    double var = temp/((rows - 2 * boundingBox )*(cols - boundingBox - cols_start_aft_cutout) - 1);
    return var;
}

void PoseEstimate::generatePose()
{
    tf::Point P1_ekf(
                pose_ekf1.position.x,
                pose_ekf1.position.y,
                pose_ekf1.position.z);

    tf::Quaternion q1_ekf(
                pose_ekf1.orientation.x,
                pose_ekf1.orientation.y,
                pose_ekf1.orientation.z,
                pose_ekf1.orientation.w);
    tf::Matrix3x3 m1_ekf(q1_ekf);
    double roll1_ekf, pitch1_ekf, yaw1_ekf;
    m1_ekf.getRPY(roll1_ekf, pitch1_ekf, yaw1_ekf);


    tf::Point P2_ekf(
                pose_ekf2.position.x,
                pose_ekf2.position.y,
                pose_ekf2.position.z);
    tf::Quaternion q2_ekf(
                pose_ekf2.orientation.x,
                pose_ekf2.orientation.y,
                pose_ekf2.orientation.z,
                pose_ekf2.orientation.w);
    tf::Matrix3x3 m2_ekf(q2_ekf);
    double roll2_ekf, pitch2_ekf, yaw2_ekf;
    m2_ekf.getRPY(roll2_ekf, pitch2_ekf, yaw2_ekf);

    double droll, dpitch, dyaw;


    droll = roll1_ekf + (1-_estimator_weight)*(roll2_ekf-roll1_ekf)+_estimator_weight*droll_FM;
    dpitch = pitch1_ekf + (1-_estimator_weight)*(pitch2_ekf-pitch1_ekf)+_estimator_weight*dpitch_FM;
    dyaw = yaw1_ekf + (1-_estimator_weight)*(yaw2_ekf-yaw1_ekf)+_estimator_weight*dyaw_FM;

    tf::Quaternion q_estimate;
    q_estimate = tf::createQuaternionFromRPY(droll,dpitch,dyaw);
    q_estimate[0] = -q_estimate[0];
    q_estimate[1] = -q_estimate[1];
    q_estimate[2] = -q_estimate[2];
    q_estimate[3] = -q_estimate[3];


    est_pose.pose.position.x = pose_ekf1.position.x + (1-_estimator_weight)*(pose_ekf2.position.x - pose_ekf1.position.x)+_estimator_weight*dx_FM;
    est_pose.pose.position.y = pose_ekf1.position.y + (1-_estimator_weight)*(pose_ekf2.position.y - pose_ekf1.position.y)+_estimator_weight*dy_FM;
    est_pose.pose.position.z = pose_ekf1.position.z + (1-_estimator_weight)*(pose_ekf2.position.z - pose_ekf1.position.z)+_estimator_weight*dz_FM;

    est_pose.pose.orientation.x = q_estimate[0];
    est_pose.pose.orientation.y = q_estimate[1];
    est_pose.pose.orientation.z = q_estimate[2];
    est_pose.pose.orientation.w = q_estimate[3];


    tf::Stamped<tf::Pose> poseEst;

    tf::poseStampedMsgToTF(est_pose, poseEst);

    tf::Pose poseHolder;
    poseHolder = camera_link_tf*poseEst;

    tf::Stamped<tf::Pose> stampedPoseHolder(poseHolder, poseEst.stamp_, poseEst.frame_id_);

    tf::poseStampedTFToMsg(stampedPoseHolder, est_pose);

    //cout << "estimated pose: " << endl << est_pose << endl;

}

void PoseEstimate::estimatePose()
{
    extract_features();

    if(initializing)
        init(im2RGB);

    else
    {
        double disp_Var = getVariance(disparity2);
        if(disp_Var > 50)
        {
            cout << " Disparity Variance = " << disp_Var << " > 50.\tRejected!" << endl;
            est_pose.pose = pose_ekf2; //just use Odom
            accepted_im_count = 0; //reset so there is always two sequenctial images
        }
        else if(accepted_im_count>1) //need two consecutive good images
        {
            //cout << "Disparity Variance = " << disp_Var << " < 5.\tAccepted" << endl;

            //Feature Matching Alignment
            //generate point clouds of matched keypoints and estimate rigid body transform between them
            acceptDecision = true;
            generate_FM_Transform(); //Will update acceptDecision here if good enough matches are not found

            if (!acceptDecision)
            {//insufficient matches to make good transform estimte
                    cout << "\tLow Feature Matches.\tRejected!" << endl;
                    est_pose.pose = pose_ekf2; //just use original odom pose
            }

            else
            {
                generatePose(); //estimate the pose using both Feature match transform and odom msgs
            }


        }
        else
        {
            accepted_im_count += 1;
            est_pose.pose = pose_ekf2; //just use original odom pose until 2 consecutive good images are available

        }
        //publish estimated pose

        pose_estimate.publish(est_pose);

        //update "previous" variables
        features1 = features2;
        im1Gray = im2Gray;
        disparity1 = disparity2;
        im1RGB = im2RGB;
        pose_ekf1 = pose_ekf2;
        keypoints3D_ROI_Points1 = keypoints3D_ROI_Points2;
        keypoints3D1 = keypoints3D2;

    }
}

void PoseEstimate::getTransform()
{
    try
    {
        tf_listener.lookupTransform(imgHeader.frame_id, est_pose.header.frame_id, ros::Time(0), camera_link_tf);
    }
    catch (tf::TransformException &ex) {
      ROS_ERROR("PoseEstimator: %s",ex.what());
      ros::Duration(1.0).sleep();
      ros::shutdown(); //Don't have a passthrough coded currently so if the transform lookup fails we want to kill this node
    }
}

void PoseEstimate::estimator_reconfig_cb(pose_estimator::EstimatorConfig& cfg, uint32_t level)
{
    _estimator_weight = cfg.estimator_weight;
}

void callback(const ImageConstPtr& rect_msg, const stereo_msgs::DisparityImageConstPtr& disp_msg, const nav_msgs::Odometry::ConstPtr& odom_msg, PoseEstimate &posee_ptr)
{
  cv_bridge::CvImagePtr rectIm_ptr;
  cv_bridge::CvImagePtr dispIm_ptr;

  try
  {
      posee_ptr.imgHeader = rect_msg->header;
      posee_ptr.est_pose.header = odom_msg->header;
      posee_ptr.pose_ekf2 = odom_msg->pose.pose;
      rectIm_ptr = cv_bridge::toCvCopy(rect_msg, image_encodings::BGR8);
      posee_ptr.im2RGB = rectIm_ptr->image;

      //Convert 32F disparity to 8U Grayscale
      const cv::Mat_<float> dmat(disp_msg->image.height, disp_msg->image.width, (float*)&disp_msg->image.data[0], disp_msg->image.step);
      //cout << dmat << endl;
      dmat.convertTo(posee_ptr.disparity2,CV_8U);
      posee_ptr.estimatePose();

//      //Display images for debug purposes
//      cv::imshow("left_rectified_image", rectIm_ptr->image);
//      cv::waitKey(30);

//      cv::imshow("Disparity Img", posee_ptr.disparity2);
//      cv::waitKey(30);

  }
  catch (cv_bridge::Exception& e)
  {
      ROS_ERROR("cv_bridge exception: %s", e.what());
  }
}



int main(int argc, char **argv){
    // required to access ros. If this gives an error, make sure you are running
    // roscore on your machine.
    ros::init(argc,argv,"PoseEstimate");
    ros::NodeHandle nh;
    ros::Duration(.1).sleep();

    PoseEstimate posee(&nh);

    ROS_INFO("Initializing Subscribers");
    message_filters::Subscriber<Image> left_rect_sub(nh, "/left_right/left_rect/image_raw", 1);
    message_filters::Subscriber<stereo_msgs::DisparityImage> disparity_sub(nh, "/left_right/left_rect/disparity", 1);
    message_filters::Subscriber<nav_msgs::Odometry> odom_sub(nh, "/pinocchio/mavros/local_position/odom", 1);

    typedef sync_policies::ApproximateTime<Image, stereo_msgs::DisparityImage, nav_msgs::Odometry> SyncPolicy;
    Synchronizer<SyncPolicy> sync(SyncPolicy(10), left_rect_sub, disparity_sub, odom_sub);
    sync.registerCallback(boost::bind(&callback, _1, _2, _3, boost::ref(posee)));


//    cv::namedWindow("left_rectified_image");
//    cv::namedWindow("Disparity Img");
//    cv::startWindowThread();

    ROS_INFO("Running");



    ros::spin();


//    cv::destroyWindow("left_rectified_image");
//    cv::destroyWindow("Disparity Img");
    return 0;
}
