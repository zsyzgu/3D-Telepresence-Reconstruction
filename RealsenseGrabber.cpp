#include "RealsenseGrabber.h"
#include "Timer.h"
#include "librealsense2/hpp/rs_sensor.hpp"
#include "librealsense2/hpp/rs_processing.hpp"
#include <mutex>

RealsenseGrabber::RealsenseGrabber()
{
	depthFilter = new DepthFilter();
	colorFilter = new ColorFilter();
	for (int i = 0; i < MAX_CAMERAS; i++) {
		decimationFilter[i] = new rs2::decimation_filter();
	}
	depthImages = new UINT16*[MAX_CAMERAS];
	colorImages = new UINT8*[MAX_CAMERAS];
	for (int i = 0; i < MAX_CAMERAS; i++) {
		colorImages[i] = new UINT8[2 * COLOR_H * COLOR_W];
	}
	colorImagesRGB = new RGBQUAD*[MAX_CAMERAS];
	for (int i = 0; i < MAX_CAMERAS; i++) {
		colorImagesRGB[i] = new RGBQUAD[COLOR_H * COLOR_W];
	}
	depthTrans = new Transformation[MAX_CAMERAS];
	depthIntrinsics = new Intrinsics[MAX_CAMERAS];
	colorIntrinsics = new Intrinsics[MAX_CAMERAS];

	rs2::context context;
	for (auto&& device : context.query_devices()) {
		enableDevice(device);
	}
}

RealsenseGrabber::~RealsenseGrabber()
{
	if (depthFilter != NULL) {
		delete depthFilter;
	}
	if (colorFilter != NULL) {
		delete colorFilter;
	}
	for (int i = 0; i < MAX_CAMERAS; i++) {
		if (decimationFilter[i] != NULL) {
			delete decimationFilter[i];
		}
	}
	if (depthImages != NULL) {
		delete[] depthImages;
	}
	if (colorImages != NULL) {
		for (int i = 0; i < MAX_CAMERAS; i++) {
			if (colorImages[i] != NULL) {
				delete colorImages[i];
			}
		}
		delete[] colorImages;
	}
	if (colorImagesRGB != NULL) {
		for (int i = 0; i < MAX_CAMERAS; i++) {
			if (colorImagesRGB[i] != NULL) {
				delete colorImagesRGB[i];
			}
		}
		delete[] colorImagesRGB;
	}
	if (depthTrans != NULL) {
		delete[] depthTrans;
	}
	if (depthIntrinsics != NULL) {
		delete[] depthIntrinsics;
	}
	if (colorIntrinsics != NULL) {
		delete[] colorIntrinsics;
	}
	for (int i = 0; i < devices.size(); i++) {
		devices[i].stop();
	}
}

void RealsenseGrabber::enableDevice(rs2::device device)
{
	std::string serialNumber(device.get_info(RS2_CAMERA_INFO_SERIAL_NUMBER));

	if (strcmp(device.get_info(RS2_CAMERA_INFO_NAME), "Platform Camera") == 0) {
		return;
	}

	rs2::config cfg;
	cfg.enable_device(serialNumber);
	cfg.enable_stream(RS2_STREAM_DEPTH, DEPTH_W * 2, DEPTH_H * 2, RS2_FORMAT_Z16, 60);
	cfg.enable_stream(RS2_STREAM_COLOR, COLOR_W, COLOR_H, RS2_FORMAT_YUYV, 60);

	rs2::pipeline pipeline;
	pipeline.start(cfg);

	devices.push_back(pipeline);

	std::vector<rs2::sensor> sensors = device.query_sensors();
	for (int i = 0; i < sensors.size(); i++) {
		if (strcmp(sensors[i].get_info(RS2_CAMERA_INFO_NAME), "Stereo Module") == 0) {
			float depth_unit = sensors[i].get_option(RS2_OPTION_DEPTH_UNITS);
			float stereo_baseline = sensors[i].get_option(RS2_OPTION_STEREO_BASELINE) * 0.001;
			convertFactors.push_back(stereo_baseline * (1 << 5) / depth_unit);
			break;
		}
	}
}

void RealsenseGrabber::convertYUVtoRGBA(UINT8* src, RGBQUAD* dst) {
	INT16 U, V;
	for (int i = 0; i < COLOR_H * COLOR_W; i++) {
		if ((i & 1) == 0) {
			U = src[i * 2 + 1];
			V = src[i * 2 + 3];
		}
		INT16 Y = src[i * 2];
		INT16 C = Y - 16;
		INT16 D = U - 128;
		INT16 E = V - 128;
		dst[i].rgbBlue = max(0, min(255, (298 * C + 409 * E + 128) >> 8));
		dst[i].rgbGreen = max(0, min(255, (298 * C - 100 * D - 208 * E + 128) >> 8));
		dst[i].rgbRed = max(0, min(255, (298 * C + 516 * D + 128) >> 8));
	}
}

int RealsenseGrabber::getRGBD(float*& depthImages_device, RGBQUAD**& colorImages, RGBQUAD*& colorImages_device, Transformation*& depthTrans, Intrinsics*& depthIntrinsics, Intrinsics*& colorIntrinsics)
{
#pragma omp parallel for
	for (int deviceId = 0; deviceId < devices.size(); deviceId++) {
		rs2::pipeline pipeline = devices[deviceId];
		rs2::frameset frameset;

		if (pipeline.poll_for_frames(&frameset) && frameset.size() > 0) {
			rs2::stream_profile depthProfile;
			rs2::stream_profile colorProfile;

			for (int i = 0; i < frameset.size(); i++) {
				rs2::frame frame = frameset[i];
				rs2::stream_profile profile = frame.get_profile();
				rs2_intrinsics intrinsics = profile.as<rs2::video_stream_profile>().get_intrinsics();

				if (profile.stream_type() == RS2_STREAM_DEPTH) {
					depthProfile = profile;
					this->depthIntrinsics[deviceId].fx = intrinsics.fx * 0.5;
					this->depthIntrinsics[deviceId].fy = intrinsics.fy * 0.5;
					this->depthIntrinsics[deviceId].ppx = intrinsics.ppx * 0.5;
					this->depthIntrinsics[deviceId].ppy = intrinsics.ppy * 0.5;
					depthFilter->setConvertFactor(deviceId, intrinsics.fx * 0.5 * convertFactors[deviceId]);
					frame = decimationFilter[deviceId]->process(frame);
					this->depthImages[deviceId] = (UINT16*)frame.get_data();
				}
				if (profile.stream_type() == RS2_STREAM_COLOR) {
					colorProfile = profile;
					this->colorIntrinsics[deviceId].fx = intrinsics.fx;
					this->colorIntrinsics[deviceId].fy = intrinsics.fy;
					this->colorIntrinsics[deviceId].ppx = intrinsics.ppx;
					this->colorIntrinsics[deviceId].ppy = intrinsics.ppy;
					std::mutex lock();
					memcpy(this->colorImages[deviceId], frame.get_data(), 2 * COLOR_W * COLOR_H * sizeof(UINT8));
				}
			}

			rs2_extrinsics extrinsics = colorProfile.get_extrinsics_to(depthProfile);
			this->depthTrans[deviceId] = Transformation(extrinsics.rotation, extrinsics.translation);
		}
	}

	for (int i = 0; i < devices.size(); i++) {
		if (this->depthImages[i] != NULL) {
			depthFilter->process(i, this->depthImages[i]);
		}
	}

	for (int i = 0; i < devices.size(); i++) {
		if (this->colorImages[i] != NULL) {
			colorFilter->process(i, this->colorImages[i]);
		}
	}

	depthImages_device = depthFilter->getCurrFrame_device();
	colorImages_device = colorFilter->getCurrFrame_device();
	depthTrans = this->depthTrans;
	depthIntrinsics = this->depthIntrinsics;
	colorIntrinsics = this->colorIntrinsics;

	return devices.size();
}

int RealsenseGrabber::getRGB(RGBQUAD**& colorImages, Intrinsics*& colorIntrinsics)
{
#pragma omp parallel for
	for (int deviceId = 0; deviceId < devices.size(); deviceId++) {
		rs2::pipeline pipeline = devices[deviceId];
		rs2::frameset frameset;
		if (pipeline.poll_for_frames(&frameset) && frameset.size() > 0) {
			for (int i = 0; i < frameset.size(); i++) {
				rs2::frame frame = frameset[i];
				rs2::stream_profile profile = frame.get_profile();
				rs2_intrinsics intrinsics = profile.as<rs2::video_stream_profile>().get_intrinsics();

				if (profile.stream_type() == RS2_STREAM_COLOR) {
					this->colorIntrinsics[deviceId].fx = intrinsics.fx;
					this->colorIntrinsics[deviceId].fy = intrinsics.fy;
					this->colorIntrinsics[deviceId].ppx = intrinsics.ppx;
					this->colorIntrinsics[deviceId].ppy = intrinsics.ppy;
					std::mutex lock();
					memcpy(this->colorImages[deviceId], frame.get_data(), 2 * COLOR_W * COLOR_H * sizeof(UINT8));
				}
			}
		}
	}

	for (int i = 0; i < devices.size(); i++) {
		if (this->colorImages[i] != NULL) {
			convertYUVtoRGBA(this->colorImages[i], this->colorImagesRGB[i]);
		}
		else {
			std::cout << "Disconnected!" << std::endl;
		}
	}

	colorImages = this->colorImagesRGB;
	colorIntrinsics = this->colorIntrinsics;
	return devices.size();
}
