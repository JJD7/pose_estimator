pcl::PointXYZRGB pose_estimator::generateUAVpos()
{
        pcl::PointXYZRGB position;
        position.x = pose_ekf2.position.x;
        position.y = pose_ekf2.position.y;
        position.z = pose_ekf2.position.z;
        uint32_t rgb = (uint32_t)255 << 16;	//red
        position.rgb = *reinterpret_cast<float*>(&rgb);

        return position;
}

void pose_estimator::generatePose()
{

}

pcl::PointXYZRGB pose_estimator::transformPoint(pcl::PointXYZRGB hexPosMAVLink, pcl::registration::TransformationEstimation<pcl::PointXYZRGB, pcl::PointXYZRGB>::Matrix4 T_SVD_matched_pts)
{
        pcl::PointXYZRGB hexPosFM;// = pcl::transformPoint(hexPosMAVLink, T_SVD_matched_pts);
        hexPosFM.x = static_cast<float> (T_SVD_matched_pts (0, 0) * hexPosMAVLink.x + T_SVD_matched_pts (0, 1) * hexPosMAVLink.y + T_SVD_matched_pts (0, 2) * hexPosMAVLink.z + T_SVD_matched_pts (0, 3));
        hexPosFM.y = static_cast<float> (T_SVD_matched_pts (1, 0) * hexPosMAVLink.x + T_SVD_matched_pts (1, 1) * hexPosMAVLink.y + T_SVD_matched_pts (1, 2) * hexPosMAVLink.z + T_SVD_matched_pts (1, 3));
        hexPosFM.z = static_cast<float> (T_SVD_matched_pts (2, 0) * hexPosMAVLink.x + T_SVD_matched_pts (2, 1) * hexPosMAVLink.y + T_SVD_matched_pts (2, 2) * hexPosMAVLink.z + T_SVD_matched_pts (2, 3));
        uint32_t rgbFM = (uint32_t)255 << 8;	//green
        hexPosFM.rgb = *reinterpret_cast<float*>(&rgbFM);

        return hexPosFM;
}

void pose_estimator::generate_tf_of_Matched_Keypoints(bool &acceptDecision)
{
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr current_img_matched_keypoints (new pcl::PointCloud<pcl::PointXYZRGB> ());
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr prior_img_matched_keypoints (new pcl::PointCloud<pcl::PointXYZRGB> ());
        current_img_matched_keypoints->is_dense = true;
        prior_img_matched_keypoints->is_dense = true;

        //find matches and create matched point clouds
        int good_matches_count = generate_Matched_Keypoints_Point_Cloud(current_img_matched_keypoints, prior_img_matched_keypoints);

        if (good_matches_count < featureMatchingThreshold)
        {
                acceptDecision = false;
        }

        //cout << "Compute Rigid Transform" << endl;
        //pcl::registration::TransformationEstimationSVD<pcl::PointXYZRGB, pcl::PointXYZRGB> te2;
        //te2.estimateRigidTransformation(*current_img_matched_keypoints, *prior_img_matched_keypoints, T_SVD_matched_pts);

        //cout << "T_SVD_Matched_Points " << endl << T_SVD_matched_pts << endl;
}


pcl::registration::TransformationEstimation<pcl::PointXYZRGB, pcl::PointXYZRGB>::Matrix4 pose_estimator::generateTmat(geometry_msgs::Pose& pose)
{
	//rotation of image plane to account for camera pitch -> x axis is towards east and y axis is towards south of image
	pcl::registration::TransformationEstimation<pcl::PointXYZRGB, pcl::PointXYZRGB>::Matrix4 r_xi;
	for (int i = 0; i < 4; i++)
		for (int j = 0; j < 4; j++)
			r_xi(i,j) = 0;
	r_xi(0,0) = r_xi(1,1) = r_xi(2,2) = 1.0;
	r_xi(3,3) = 1.0;
	r_xi(1,1) = cos(theta_xi);
	r_xi(1,2) = -sin(theta_xi);
	r_xi(2,1) = sin(theta_xi);
	r_xi(2,2) = cos(theta_xi);
	
	//rotation of image plane to account for camera roll-> x axis is towards east and y axis is towards south of image
	pcl::registration::TransformationEstimation<pcl::PointXYZRGB, pcl::PointXYZRGB>::Matrix4 r_yi;
	for (int i = 0; i < 4; i++)
		for (int j = 0; j < 4; j++)
			r_yi(i,j) = 0;
	r_yi(0,0) = r_yi(1,1) = r_yi(2,2) = 1.0;
	r_yi(3,3) = 1.0;
	r_yi(0,0) = cos(theta_yi);
	r_yi(0,2) = sin(theta_yi);
	r_yi(2,0) = -sin(theta_yi);
	r_yi(2,2) = cos(theta_yi);
	
	//rotation of image plane to invert y axis and designate all z points negative -> now x axis is towards east and y axis is towards north of image and image
	pcl::registration::TransformationEstimation<pcl::PointXYZRGB, pcl::PointXYZRGB>::Matrix4 r_invert_i;
	for (int i = 0; i < 4; i++)
		for (int j = 0; j < 4; j++)
			r_invert_i(i,j) = 0;
	r_invert_i(3,3) = 1.0;
	////invert y and z
	r_invert_i(0,0) = 1.0;	//x
	r_invert_i(1,1) = -1.0;	//y
	r_invert_i(2,2) = -1.0;	//z
	pcl::registration::TransformationEstimation<pcl::PointXYZRGB, pcl::PointXYZRGB>::Matrix4 r_invert_y;
	for (int i = 0; i < 4; i++)
		for (int j = 0; j < 4; j++)
			r_invert_y(i,j) = 0;
	r_invert_y(3,3) = 1.0;
	r_invert_y(0,0) = 1.0;	//x
	r_invert_y(1,1) = -1.0;	//y
	r_invert_y(2,2) = 1.0;	//z
	pcl::registration::TransformationEstimation<pcl::PointXYZRGB, pcl::PointXYZRGB>::Matrix4 r_invert_z;
	for (int i = 0; i < 4; i++)
		for (int j = 0; j < 4; j++)
			r_invert_z(i,j) = 0;
	r_invert_z(3,3) = 1.0;
	r_invert_z(0,0) = 1.0;	//x
	r_invert_z(1,1) = 1.0;	//y
	r_invert_z(2,2) = -1.0;	//z
	
	//translation image plane to hexacopter
	pcl::registration::TransformationEstimation<pcl::PointXYZRGB, pcl::PointXYZRGB>::Matrix4 t_hi;
	for (int i = 0; i < 4; i++)
		for (int j = 0; j < 4; j++)
			t_hi(i,j) = 0;
	t_hi(0,0) = t_hi(1,1) = t_hi(2,2) = 1.0;
	t_hi(0,3) = trans_x_hi;
	t_hi(1,3) = trans_y_hi;
	t_hi(2,3) = trans_z_hi;
	t_hi(3,3) = 1.0;
	
	//rotation to flip x and y axis-> now x axis is towards north and y axis is towards east of hexacopter
	pcl::registration::TransformationEstimation<pcl::PointXYZRGB, pcl::PointXYZRGB>::Matrix4 r_flip_xy;
	for (int i = 0; i < 4; i++)
		for (int j = 0; j < 4; j++)
			r_flip_xy(i,j) = 0;
	r_flip_xy(3,3) = 1.0;
	//flip x and y
	r_flip_xy(1,0) = 1.0;	//x
	r_flip_xy(0,1) = 1.0;	//y
	r_flip_xy(2,2) = 1.0;	//z
	
	////rotate to invert y axis I dont know why this correction works!
	//pcl::registration::TransformationEstimation<pcl::PointXYZRGB, pcl::PointXYZRGB>::Matrix4 r_invert_y;
	//for (int i = 0; i < 4; i++)
	//	for (int j = 0; j < 4; j++)
	//		r_invert_y(i,j) = 0;
	//r_invert_y(3,3) = 1.0;
	////invert y
	//r_invert_y(0,0) = 1.0;	//x
	//r_invert_y(1,1) = -1.0;	//y
	//r_invert_y(2,2) = 1.0;	//z
	
	
	//converting hexacopter quaternion to rotation matrix
        double tx = pose.position.x;
        double ty = pose.position.y;
        double tz = pose.position.z;
        double qx = pose.orientation.x;
        double qy = pose.orientation.y;
        double qz = pose.orientation.z;
        double qw = pose.orientation.w;
	
	double sqw = qw*qw;
	double sqx = qx*qx;
	double sqy = qy*qy;
	double sqz = qz*qz;
	
	if(sqw + sqx + sqy + sqz < 0.99 || sqw + sqx + sqy + sqz > 1.01)
		throw "Exception: Sum of squares of quaternion values should be 1! i.e., quaternion should be homogeneous!";
	
	Mat rot = Mat::zeros(cv::Size(3, 3), CV_64FC1);
	
	rot.at<double>(0,0) = sqx - sqy - sqz + sqw; // since sqw + sqx + sqy + sqz =1
	rot.at<double>(1,1) = -sqx + sqy - sqz + sqw;
	rot.at<double>(2,2) = -sqx - sqy + sqz + sqw;

	double tmp1 = qx*qy;
	double tmp2 = qz*qw;
	rot.at<double>(0,1) = 2.0 * (tmp1 + tmp2);
	rot.at<double>(1,0) = 2.0 * (tmp1 - tmp2);

	tmp1 = qx*qz;
	tmp2 = qy*qw;
	rot.at<double>(0,2) = 2.0 * (tmp1 - tmp2);
	rot.at<double>(2,0) = 2.0 * (tmp1 + tmp2);

	tmp1 = qy*qz;
	tmp2 = qx*qw;
	rot.at<double>(1,2) = 2.0 * (tmp1 + tmp2);
	rot.at<double>(2,1) = 2.0 * (tmp1 - tmp2);
	
	rot = rot.t();
	
	//rotation to orient hexacopter coordinates to world coordinates
	pcl::registration::TransformationEstimation<pcl::PointXYZRGB, pcl::PointXYZRGB>::Matrix4 r_wh;
	for (int i = 0; i < 3; i++)
		for (int j = 0; j < 3; j++)
			r_wh(i,j) = rot.at<double>(i,j);
	r_wh(3,0) = r_wh(3,1) = r_wh(3,2) = r_wh(0,3) = r_wh(1,3) = r_wh(2,3) = 0.0;
	r_wh(3,3) = 1.0;
	
	//translation to translate hexacopter coordinates to world coordinates
	pcl::registration::TransformationEstimation<pcl::PointXYZRGB, pcl::PointXYZRGB>::Matrix4 t_wh;
	for (int i = 0; i < 4; i++)
		for (int j = 0; j < 4; j++)
			t_wh(i,j) = 0;
	t_wh(0,0) = t_wh(1,1) = t_wh(2,2) = 1.0;
	t_wh(0,3) = tx;
	t_wh(1,3) = ty;
	t_wh(2,3) = tz;
	t_wh(3,3) = 1.0;
	
	//translate hexacopter take off position to base station antenna location origin : Translation: [1.946, 6.634, -1.006]
        const double antenna_takeoff_offset_x = 0;
        const double antenna_takeoff_offset_y = 0;
        const double antenna_takeoff_offset_z = 0;
	pcl::registration::TransformationEstimation<pcl::PointXYZRGB, pcl::PointXYZRGB>::Matrix4 t_ow;
	for (int i = 0; i < 4; i++)
		for (int j = 0; j < 4; j++)
			t_ow(i,j) = 0;
	t_ow(0,0) = t_ow(1,1) = t_ow(2,2) = 1.0;
	t_ow(0,3) = - antenna_takeoff_offset_x;
	t_ow(1,3) = - antenna_takeoff_offset_y;
	t_ow(2,3) = - antenna_takeoff_offset_z;
	t_ow(3,3) = 1.0;
		
	pcl::registration::TransformationEstimation<pcl::PointXYZRGB, pcl::PointXYZRGB>::Matrix4 t_mat = t_wh * r_wh * r_invert_y * r_flip_xy * t_hi * r_invert_i * r_yi * r_xi;

	
	return t_mat;
}
