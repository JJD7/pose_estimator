#include "pose_estimator.h"
#include "functions.cpp"

const int MAX_FEATURES = 1000;
const float GOOD_MATCH_PERCENT = 0.15f;
bool initializing = true;

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


void pose_estimator::createImgPtCloud(Mat &im, pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloudrgb)
{

    //cout << " Pt Cloud #" << accepted_img_index << flush;
    cloudrgb->is_dense = true;

    Mat dispImg;
    dispImg = disparity2;

    if(blur_kernel > 1)
    {
        //blur the disparity image to remove noise
        Mat disp_img_blurred;
        bilateralFilter ( dispImg, disp_img_blurred, blur_kernel, blur_kernel*2, blur_kernel/2 );
        //medianBlur ( disp_img, disp_img_blurred, blur_kernel );
        dispImg = disp_img_blurred;
    }

    vector<KeyPoint> keypoints = features2.keypoints;
    Mat rgb_image = im2RGB;

    cv::Mat_<double> vec_tmp(4,1);

    //when jump_pixels == 1, all keypoints will be already included later as we will take in all points
    //with jump_pixels == 0, we only want to take in keypoints
    if (jump_pixels != 1)
    {
        for (int i = 0; i < keypoints.size(); i++)
        {
            int x = keypoints[i].pt.x, y = keypoints[i].pt.y;
            if (x >= cols_start_aft_cutout && x < cols - boundingBox && y >= boundingBox && y < rows - boundingBox)
            {
                double disp_val = 0;
                disp_val = (double)dispImg.at<uchar>(y,x);

                if (disp_val > minDisparity)
                {
                    //reference: https://stackoverflow.com/questions/22418846/reprojectimageto3d-in-opencv
                    vec_tmp(0)=x; vec_tmp(1)=y; vec_tmp(2)=disp_val; vec_tmp(3)=1;
                    vec_tmp = Q*vec_tmp;
                    vec_tmp /= vec_tmp(3);

                    pcl::PointXYZRGB pt_3drgb;
                    pt_3drgb.x = (float)vec_tmp(0);
                    pt_3drgb.y = (float)vec_tmp(1);
                    pt_3drgb.z = (float)vec_tmp(2);
                    Vec3b color = rgb_image.at<Vec3b>(Point(x, y));

                    uint32_t rgb = ((uint32_t)color[2] << 16 | (uint32_t)color[1] << 8 | (uint32_t)color[0]);
                    pt_3drgb.rgb = *reinterpret_cast<float*>(&rgb);

                    cloudrgb->points.push_back(pt_3drgb);
                    //cout << pt_3d << endl;
                }
            }
        }
    }
    if (jump_pixels > 0)
    {
        for (int y = boundingBox; y < rows - boundingBox;)
        {
            for (int x = cols_start_aft_cutout; x < cols - boundingBox;)
            {
                double disp_val = 0;
                disp_val = (double)dispImg.at<uchar>(y,x);

                if (disp_val > minDisparity)
                {
                    //reference: https://stackoverflow.com/questions/22418846/reprojectimageto3d-in-opencv
                    vec_tmp(0)=x; vec_tmp(1)=y; vec_tmp(2)=disp_val; vec_tmp(3)=1;
                    vec_tmp = Q*vec_tmp;
                    vec_tmp /= vec_tmp(3);

                    pcl::PointXYZRGB pt_3drgb;
                    pt_3drgb.x = (float)vec_tmp(0);
                    pt_3drgb.y = (float)vec_tmp(1);
                    pt_3drgb.z = (float)vec_tmp(2);
                    Vec3b color = rgb_image.at<Vec3b>(Point(x, y));

                    uint32_t rgb = ((uint32_t)color[2] << 16 | (uint32_t)color[1] << 8 | (uint32_t)color[0]);
                    pt_3drgb.rgb = *reinterpret_cast<float*>(&rgb);

                    cloudrgb->points.push_back(pt_3drgb);
                    //cout << pt_3d << endl;
                }
                x += jump_pixels;
            }
            y += jump_pixels;
        }
    }
}

void pose_estimator::extract_features(Mat &im)
{

    // Convert images to grayscale
    im2RGB = im;
    cvtColor(im, im2Gray, CV_BGR2GRAY);

    finder = makePtr<OrbFeaturesFinder>();
    (*finder)(im2RGB, features2);

    //cuda::GpuMat descriptor(features2.descriptors);
    //gpu_descriptors2 = descriptor;

    vector<KeyPoint> keypoints = features2.keypoints;

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr keypoints3dptcloud (new pcl::PointCloud<pcl::PointXYZRGB> ());
    keypoints3dptcloud->is_dense = true;
    keypoints3D2 = keypoints3dptcloud;
    vector<bool> pointsInROIVec;

    int good = 0, bad = 0;
    for (int i = 0; i < keypoints.size(); i++)
    {
            double disp_value;
            disp_value = disparity2.at<char>(keypoints[i].pt.y, keypoints[i].pt.x);

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
    //cout << " g" << good << "/b" << bad << flush;
    cout << " g" << good << "/b" << bad << flush <<endl;
    keypoints3D_ROI_Points2 = pointsInROIVec;
}

int pose_estimator::generate_Matched_Keypoints_Point_Cloud
(pcl::PointCloud<pcl::PointXYZRGB>::Ptr &current_img_matched_keypoints,
pcl::PointCloud<pcl::PointXYZRGB>::Ptr &fitted_cloud_matched_keypoints)
{
    cout << " matched with_imgs/matches";

    int good_matched_imgs_this_src = 0;
    int good_matches_count = 0;

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr keypoints3D_src = keypoints3D2;
    vector<bool> pointsInROIVec_src = keypoints3D_ROI_Points2;


    //reference https://stackoverflow.com/questions/44988087/opencv-feature-matching-match-descriptors-to-knn-filtered-keypoints
    //reference https://github.com/opencv/opencv/issues/6130
    //reference http://study.marearts.com/2014/07/opencv-study-orb-gpu-feature-extraction.html
    //reference https://docs.opencv.org/3.1.0/d6/d1d/group__cudafeatures2d.html

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

    if(good_matches.size() < featureMatchingThreshold/2)	//less number of matches.. don't bother working on this one. good matches are around 200-500
        cout << "Insufficient number of feature matches for estimateing Feature Matched Pose" << endl;
        // TO DO: Reject pose_estimate. Need to just use mavlink pose in this case

    good_matched_imgs_this_src++;
    good_matches_count += good_matches.size();

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

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_current_t_temp (new pcl::PointCloud<pcl::PointXYZRGB> ());
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_prior_t_temp (new pcl::PointCloud<pcl::PointXYZRGB> ());

    //pcl::transformPointCloud(*cloud_current_temp, *cloud_current_t_temp, t_mat_MAVLink2);

    pcl::transformPointCloud(*cloud_prior_temp, *cloud_prior_t_temp, t_mat_FeatureMatched1);

    current_img_matched_keypoints->insert(current_img_matched_keypoints->end(),cloud_current_t_temp->begin(),cloud_current_t_temp->end());
    fitted_cloud_matched_keypoints->insert(fitted_cloud_matched_keypoints->end(),cloud_prior_t_temp->begin(),cloud_prior_t_temp->end());

    return good_matches_count;
}

void pose_estimator::init(Mat &im)
{
    ROS_INFO("initalizing tracker");
    rows = im.rows;
    cols = im.cols;
    cols_start_aft_cutout = (int)(cols/cutout_ratio);

    features1 = features2;
    im1Gray = im2Gray;
    im1RGB = im2RGB;
    //orb_img1 = orb_img2;
    initializing = false;
    //gpu_descriptors1 = gpu_descriptors2;
    pose_ekf1 = pose_ekf2;

}

double pose_estimator::getMean(Mat disp_img)
{
        double sum = 0.0;
        for (int y = boundingBox; y < rows - boundingBox; ++y)
        {
                for (int x = cols_start_aft_cutout; x < cols - boundingBox; ++x)
                {
                    double disp_val = 0;
                    disp_val = (double)disp_img.at<uchar>(y,x);

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
                    disp_val = (double)disp_img.at<uchar>(y,x);

                    if (disp_val > minDisparity)
                        temp += (disp_val-mean)*(disp_val-mean);
                }
        }
        double var = temp/((rows - 2 * boundingBox )*(cols - boundingBox - cols_start_aft_cutout) - 1);
        return var;
}

void pose_estimator::estimatePose()
{
    if(initializing)
        init(im2RGB);

    else
    {
        double disp_Var = getVariance(disparity2);
        if(disp_Var > 5)
        {
            cout << " Disparity Variance = " << disp_Var << " > 5.\tRejected!" << endl;
            est_pose = pose_ekf2; //just use Odom
            return;
        }
        else
        {
            cout << "Disparity Variance = " << disp_Var << " < 5.\tAccepted" << endl;
        }
        //pcl::registration::TransformationEstimation<pcl::PointXYZRGB, pcl::PointXYZRGB>::Matrix4 t_mat_MAVLink = generateTmat(pose_ekf2);
        //publish estimated pose
        est_pose = pose_ekf2;
        pose_estimate.publish(est_pose);
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
      const Image& dimage = disp_msg->image;
      const cv::Mat_<float> dmat(dimage.height, dimage.width, (float*)&dimage.data[0], dimage.step);
      double min, max;
      cv::minMaxLoc(dmat, &min, &max);

      if (min!=max) //dont divide by zero
          dmat.convertTo(posee_ptr.disparity2,CV_8U,255.0/max-min);

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
    ros::init(argc,argv,"pose_estimator");
    ros::NodeHandle nh;
    ros::Duration(.1).sleep();

    pose_estimator posee(&nh);

    ROS_INFO("Initializing Subscribers");
    message_filters::Subscriber<Image> left_rect_sub(nh, "/left_right/left_rect/image_raw", 1);
    message_filters::Subscriber<stereo_msgs::DisparityImage> disparity_sub(nh, "/left_right/left_rect/disparity", 1);
    message_filters::Subscriber<nav_msgs::Odometry> odom_sub(nh, "pinocchio/mavros/local_position/odom", 1);

    typedef sync_policies::ApproximateTime<Image, stereo_msgs::DisparityImage, nav_msgs::Odometry> SyncPolicy;
    Synchronizer<SyncPolicy> sync(SyncPolicy(10), left_rect_sub, disparity_sub, odom_sub);
    sync.registerCallback(boost::bind(&callback, _1, _2, _3, posee));


//    cv::namedWindow("left_rectified_image");
//    cv::namedWindow("Disparity Img");
//    cv::startWindowThread();

    ROS_INFO("Running");



    ros::spin();


//    cv::destroyWindow("left_rectified_image");
//    cv::destroyWindow("Disparity Img");
    return 0;
}
