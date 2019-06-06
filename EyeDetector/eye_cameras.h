#ifndef EYE_CAMERAS_H
#define EYE_CAMERAS_H

#include <iostream>
#include <string>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

namespace eye_tracker
{
	/**
	* @class CameraUndistorter
	* @brief Image undistortion class. This class keeps undistortion map for efficiency
	*/
	class CameraUndistorter
	{
	public:
		CameraUndistorter(const cv::Mat &K, const cv::Vec<double, 8> &distCoeffs)
			: K0_(K.clone()), distCoeffs_(distCoeffs)
		{
			std::cout << "Intrinsic (matrix): " << K0_ << std::endl;
			std::cout << "Intrinsic (distortion): " << distCoeffs_ << std::endl;

		}
		~CameraUndistorter() {
		}
		void init_maps(const cv::Size &s) {
			cv::initUndistortRectifyMap(K0_, distCoeffs_, cv::Mat(), K0_, s, CV_32FC1, mapx_, mapy_);
		}
		void undistort(const cv::Mat &in, cv::Mat &out) {
			if (is_dist_map_initialized_ == false) {
				init_maps(in.size());
				is_dist_map_initialized_ = true;
			}
			cv::remap(in, out, mapx_, mapy_, cv::INTER_LINEAR);
		}
	protected:
		// Local variables initialized at the constructor
		cv::Mat K0_;
		cv::Vec<double, 8> distCoeffs_; // (k1 k2 p1 p2 [k3 [k4 k5 k6]])
		bool is_dist_map_initialized_ = false;
		cv::Mat mapx_, mapy_;
	private:
		// Prevent copying
	};

} // namespace
#endif // IRIS_DETECTOR_IR_H