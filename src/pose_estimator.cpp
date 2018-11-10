#include "pose_estimator.h"

const int MAX_FEATURES = 1000;
const float GOOD_MATCH_PERCENT = 0.15f;
bool initializing = true;
double accepted_im_count =0;
float a = 0.5;
float b = 1-a;
bool acceptDecision = true;

pose_estimator::pose_estimator(ros::NodeHandle* nodehandle):nh_(*nodehandle)
{ // constructor
    ROS_INFO("in class constructor of pose_estimator");
    readCalibFile();

    ROS_INFO("Initializing Publishers");
    pose_estimate = nh_.advertise<geometry_msgs::Pose>("estimated_pose",1);
}

void pose_estimator::readCalibFile()
{
        ROS_INFO("Loading Calibration File");
        cv::FileStorage fs(calib_file_Dir + calib_file, cv::FileStorage::READ);
        fs["Q"] >> Q;
        if(Q.empty())
                throw "Exception: could not read Q matrix";
        fs.release();
        ROS_INFO("Loaded Camera Q matrix as: ");
        std::cout << Q << endl;
}

void pose_estimator::extract_features()
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

void pose_estimator::generate_FM_Transform()
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

    icp.setInputSource(cloud_prior_temp);
    icp.setInputTarget(cloud_current_temp);
    pcl::PointCloud<pcl::PointXYZRGB> Final;
    icp.align(Final);
    std::cout << "has converged:" << icp.hasConverged() << " score: " <<
    icp.getFitnessScore() << std::endl;
    //std::cout << icp.getFinalTransformation() << std::endl;

    if (good_matches.size() < featureMatchingThreshold)
    {
            acceptDecision = false;
    }

}

void pose_estimator::init(Mat &im)
{
    ROS_INFO("initalizing tracker");
    rows = im.rows;
    cols = im.cols;
    cols_start_aft_cutout = (int)(cols/cutout_ratio);
    initializing = false;
}

double pose_estimator::getMean(Mat disp_img)
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

double pose_estimator::getVariance(Mat disp_img)
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

void pose_estimator::generatePose()
{
    tf::Pose Pose1;
    tf::Pose Pose2;
    tf::poseMsgToTF(pose_ekf1,Pose1);
    tf::poseMsgToTF(pose_ekf2,Pose2);
    tf::Pose diff_FCU = Pose1.inverseTimes(Pose2);// Pose1.inverse() * Pose2;
    tf::Pose Pose_Estimate = Pose1*diff_FCU;

    tf::poseTFToMsg(Pose_Estimate,est_pose);
    cout << "estimated pose: " << endl << est_pose;
    //cout << "difference in pose " << endl << diff_FCU << endl;
    //pose = pose_ekf2.pose;
}

void pose_estimator::estimatePose()
{
    extract_features();

    if(initializing)
        init(im2RGB);

    else
    {
        double disp_Var = getVariance(disparity2);
        if(disp_Var > 50)
        {
            //cout << " Disparity Variance = " << disp_Var << " > 5.\tRejected!" << endl;
            est_pose = pose_ekf2; //just use Odom
            accepted_im_count = 0; //reset so there is always two sequenctial images
        }
        else if(accepted_im_count>1) //need two consecutive good images
        {
            //cout << "Disparity Variance = " << disp_Var << " < 5.\tAccepted" << endl;

            //Feature Matching Alignment
            //generate point clouds of matched keypoints and estimate rigid body transform between them
            acceptDecision = true;
            generate_FM_Transform();

            if (!acceptDecision)
            {//rejected point -> no matches found
                    cout << "\tLow Feature Matches.\tRejected!" << endl;
            }

            generatePose();
            est_pose = pose_ekf2;
        }
        else
        {
            accepted_im_count += 1;

        }
        //publish estimated pose

        pose_estimate.publish(est_pose);

        features1 = features2;
        im1Gray = im2Gray;
        disparity1 = disparity2;
        im1RGB = im2RGB;
        pose_ekf1 = pose_ekf2;
        keypoints3D_ROI_Points1 = keypoints3D_ROI_Points2;
        keypoints3D1 = keypoints3D2;

    }
}


void callback(const ImageConstPtr& rect_msg, const stereo_msgs::DisparityImageConstPtr& disp_msg, const nav_msgs::Odometry::ConstPtr& odom_msg, pose_estimator &posee_ptr)
{
  cv_bridge::CvImagePtr rectIm_ptr;
  cv_bridge::CvImagePtr dispIm_ptr;

  try
  {
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

      cv::imshow("Disparity Img", posee_ptr.disparity2);
      cv::waitKey(30);

  }
  catch (cv_bridge::Exception& e)
  {
      ROS_ERROR("cv_bridge exception: %s", e.what());
  }
}



int main(int argc, char **argv){
    // required to access ros. If this gives an error, make sure you are running
    // roscore on your machine.
    ros::init(argc,argv,"pose_estimator");
    ros::NodeHandle nh;
    ros::Duration(.1).sleep();

    pose_estimator posee(&nh);

    ROS_INFO("Initializing Subscribers");
    message_filters::Subscriber<Image> left_rect_sub(nh, "/left_right/left_rect/image_raw", 1);
    message_filters::Subscriber<stereo_msgs::DisparityImage> disparity_sub(nh, "/left_right/left_rect/disparity", 1);
    message_filters::Subscriber<nav_msgs::Odometry> odom_sub(nh, "/mavros/local_position/odom", 1);

    typedef sync_policies::ApproximateTime<Image, stereo_msgs::DisparityImage, nav_msgs::Odometry> SyncPolicy;
    Synchronizer<SyncPolicy> sync(SyncPolicy(10), left_rect_sub, disparity_sub, odom_sub);
    sync.registerCallback(boost::bind(&callback, _1, _2, _3, posee));


//    cv::namedWindow("left_rectified_image");
    cv::namedWindow("Disparity Img");
    cv::startWindowThread();

    ROS_INFO("Running");



    ros::spin();


//    cv::destroyWindow("left_rectified_image");
    cv::destroyWindow("Disparity Img");
    return 0;
}
