#include "tracker.h"

constexpr int CAM_NUM = 2;

#ifndef PYTHONLIB
int main(int argc, char *argv[]) {

	// for debug
	std::vector<cv::VideoCapture> vid(CAM_NUM);
	vid[0].open("D:/Developing/sawai/vs/3DEyeTracker_wrapper/docs/Right.avi");
	vid[1].open("D:/Developing/sawai/vs/3DEyeTracker_wrapper/docs/Left.avi");
	std::vector<cv::Mat> images(CAM_NUM);

	Tracker* trk = new Tracker(CAM_NUM);

	// Main loop
	const char TERMINATE = 27;  // ESC key
	bool is_run = true;
	while (is_run) {

		// Fetch key input
		char key = 0;
		key = cv::waitKey(1);
		switch (key) {
		case TERMINATE:
			is_run = false;
			break;
		}

		// Fetch images
		for (size_t cam = 0; cam < CAM_NUM; cam++) {
			vid[cam] >> images[cam];
		}
		// Process each camera images
		for (size_t cam = 0; cam < CAM_NUM; cam++) {
			trk->detect(images[cam], cam, key);
		} // For each cameras

	}// Main capture loop

	free(trk);
	return 0;

}
#endif

Tracker::Tracker(int cam_num)
	: camera_undistorters(cam_num),
	window_names(cam_num),
	eye_model_updaters(cam_num)
{
	if (ubitrack_calib_text_reader.read(calib_path) == false) {
		std::cout << "Calibration file onpen error: " << calib_path << std::endl;
		exit(EXIT_FAILURE);
	}
	ubitrack_calib_text_reader.data_.get_parameters_opencv_default(K, distCoeffs);
	double focal_length = (K.at<double>(0, 0) + K.at<double>(1, 1))*0.5;
	eye_model_updaters[0] = std::make_unique<eye_tracker::EyeModelUpdater>(focal_length, 5, 0.5);
	camera_undistorters[0] = std::make_unique<eye_tracker::CameraUndistorter>(K, distCoeffs);
	if (cam_num > 1) {
		eye_model_updaters[1] = std::make_unique<eye_tracker::EyeModelUpdater>(focal_length, 5, 0.5);
		camera_undistorters[1] = std::make_unique<eye_tracker::CameraUndistorter>(K, distCoeffs);
	}
	window_names = { "Right", "Left" };
	pupilFitter.setDebug(false);

	ppl = (double*)malloc((size_t)sizeof(double) * 3);
	eybl = (double*)malloc((size_t)sizeof(double) * 3);

	std::cout << "Detector init done" << std::endl;
}

void Tracker::detect(cv::Mat& image, int cam, char key)
{
	cv::Mat &img = image;
	if (img.empty()) {
		std::cout << "Image empty" << std::endl;
		return;
	}

	// Undistort a captured tmpimg
	camera_undistorters[cam]->undistort(img, img);

	cv::Mat img_rgb = img.clone();
	cv::Mat img_grey;

	switch (key) {
	case 'r':
		eye_model_updaters[cam]->reset();
		break;
	case 'p':
		eye_model_updaters[cam]->add_fitter_max_count(10);
		break;
	default:
		break;
	}

	// 2D ellipse detection
	std::vector<cv::Point2f> inlier_pts;
	cv::cvtColor(img, img_grey, CV_RGB2GRAY);
	cv::RotatedRect rr_pf;
	bool is_pupil_found = pupilFitter.pupilAreaFitRR(img_grey, rr_pf, inlier_pts);

	singleeyefitter::Ellipse2D<double> el = singleeyefitter::toEllipse<double>(eye_tracker::toImgCoordInv(rr_pf, img, 1.0));

	// 3D eye pose estimation
	bool is_reliable = false;
	bool is_added = false;
	const bool FORCE_ADD = false;
	const double REL_TREATH = 0.5;// 0.96;
	double ellipse_reliability = 0.0; /// Reliability of a detected 2D ellipse based on 3D eye model
	if (is_pupil_found) {
		if (eye_model_updaters[cam]->is_model_built()) {
			ellipse_reliability = eye_model_updaters[cam]->compute_reliability(img, el, inlier_pts);
			is_reliable = (ellipse_reliability > REL_TREATH);
		}
		else {
			is_added = eye_model_updaters[cam]->add_observation(img_grey, el, inlier_pts, FORCE_ADD);
		}
	}

	// Visualize results
	// 2D pupil
	if (is_pupil_found) {
		cv::ellipse(img_rgb, rr_pf, cv::Vec3b(255, 128, 0), 1);
	}

	// 3D eye ball
	if (eye_model_updaters[cam]->is_model_built()) {
		//cv::putText(img, "Reliability: " + std::to_string(ellipse_reliability), cv::Point(30, 440), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 128, 255), 1);
		
		// Unproject the current 2D ellipse observation to a 3D disk
		singleeyefitter::EyeModelFitter::Circle curr_circle = eye_model_updaters[cam]->unproject(img, el, inlier_pts);
		ppl = curr_circle.centre.data();
		eybl = eye_model_updaters[cam]->fitter().eye.centre.data();
		//::cout << "pupil: " << curr_circle.centre << std::endl << "eyeball: " << eye_model_updaters[cam]->fitter().eye.centre << std::endl;
		
		if (is_reliable) {
			eye_model_updaters[cam]->render(img_rgb, curr_circle);
		}
	}
	else {
		eye_model_updaters[cam]->render_status(img_rgb);
		//cv::putText(img, "Sample #: " + std::to_string(eye_model_updaters[cam]->fitter_count()) + "/" + std::to_string(eye_model_updaters[cam]->fitter_end_count()),
		//	cv::Point(30, 440), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 128, 255), 2);
	}

	cv::imshow(window_names[cam], img_rgb);
}

int Tracker::dummy(int h, int w, int* arr) {
	if (h > 1) {
		return arr[w];
	}
	else {
		return -1;
	}
}