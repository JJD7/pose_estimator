#include "pose_estimator.h"

const int MAX_FEATURES = 1000;
const float GOOD_MATCH_PERCENT = 0.15f;
bool initializing = true;

pose_estimator::pose_estimator(ros::NodeHandle* nodehandle):nh_(*nodehandle)
{ // constructor
    ROS_INFO("in class constructor of pose_estimator");
    readCalibFile();
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

    cuda::GpuMat descriptor(features2.descriptors);
    gpu_descriptors2 = descriptor;

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

void pose_estimator::init(Mat &im)
{
    ROS_INFO("initalizing tracker");
    rows = im.rows;
    cols = im.cols;
    cols_start_aft_cutout = (int)(cols/cutout_ratio);

    features1 = features2;
    im1Gray = im2Gray;
    im1RGB = im2RGB;
    orb_img1 = orb_img2;
    initializing = false;
    gpu_descriptors1 = gpu_descriptors2;

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

void imageCallback(const sensor_msgs::ImageConstPtr& msg, pose_estimator &posee_ptr)
{
    cv_bridge::CvImagePtr cv_ptr;

    try
    {
        cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
        cv::imshow("left_rectified_image", cv_ptr->image);
        cv::waitKey(30);
    }
    catch (cv_bridge::Exception& e)
    {
        ROS_ERROR("cv_bridge exception: %s", e.what());
    }

    //extract_features(cv_ptr->image, posee_ptr);
}

//void disparityImageCallback(const stereo_msgs::DisparityImage& msg)
//{
//    cv_bridge::CvImagePtr cv_ptr;

//    try
//    {
//        cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
//        cv::imshow("left_rectified_image", cv_ptr->image);
//        cv::waitKey(30);
//    }
//    catch (cv_bridge::Exception& e)
//    {
//        ROS_ERROR("cv_bridge exception: %s", e.what());
//    }
//}

int main(int argc, char **argv){
    // required to access ros. If this gives an error, make sure you are running
    // roscore on your machine.
    ros::init(argc,argv,"pose_estimator");
    ros::NodeHandle nh;
    ros::Duration(.1).sleep();

    pose_estimator posee(&nh);

    cv::namedWindow("left_rectified_image");
    cv::startWindowThread();
    ROS_INFO("Initializing Subscribers");
    image_transport::ImageTransport it(nh);
    image_transport::Subscriber leftRect_image= it.subscribe("/left_right/left_rect/image_raw", 1, boost::bind(imageCallback, _1, posee));

//    ros::Subscriber sub = nh.subscribe("/left_right/left_rect/disparity", 1, disparityImageCallback);

    ros::Publisher pose_estimate;
    ROS_INFO("Initializing Publishers");
    pose_estimate = nh.advertise<geometry_msgs::Pose>("estimated_pose",1);


    ROS_INFO("Running");


    ros::spin();


    cv::destroyWindow("left_rectified_image");
    return 0;
}
