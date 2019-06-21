#define PYTHONLIB 0
#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <sstream>

#include "ubitrack_util.h" // claibration file handlers
#include <boost/foreach.hpp>
#include <boost/filesystem.hpp>
#include <boost/filesystem/path.hpp>
#include <boost/filesystem/fstream.hpp>
#include <boost/thread.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/photo/photo.hpp>

#include "eye_cameras.h"
#include "pupilFitter.h" // 2D pupil detector
#include "eye_model_updater.h" // 3D model builder

class Tracker
{
public:
	Tracker(int cam_num);
	void detect(cv::Mat& image, int cam, char key);

	int dummy(int w, int h, int* arr);

	std::string calib_path = { "D:/Developing/sawai/vs/EyeDetector/data/cameraintrinsics_eye.txt" };
	eye_tracker::UbitrackTextReader<eye_tracker::Caib> ubitrack_calib_text_reader;
	cv::Mat K; // Camera intrinsic matrix in OpenCV format
	cv::Vec<double, 8> distCoeffs; // (k1 k2 p1 p2 [k3 [k4 k5 k6]]) // k: radial, p: tangential

	// Variables that handle camera setups
	std::vector<std::unique_ptr<eye_tracker::CameraUndistorter>> camera_undistorters; // Camera undistorters
	std::vector<std::string> window_names;                                            // Window names
	std::vector<std::unique_ptr<eye_tracker::EyeModelUpdater>> eye_model_updaters;    // 3D eye models
	PupilFitter pupilFitter;

	const double *ppl, *eybl;
};
