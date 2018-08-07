#include "RealsenseGrabber.h"
#include "Timer.h"
#include "Configuration.h"
#include "librealsense2/hpp/rs_sensor.hpp"
#include "librealsense2/hpp/rs_processing.hpp"

RealsenseGrabber::RealsenseGrabber()
{
	depthFilter = new DepthFilter();
	colorFilter = new ColorFilter();
	alignColorMap = new AlignColorMap();
	depthImages = new UINT16*[MAX_CAMERAS];
	colorImages = new UINT8*[MAX_CAMERAS];
	for (int i = 0; i < MAX_CAMERAS; i++) {
		depthImages[i] = new UINT16[DEPTH_H * DEPTH_W];
		colorImages[i] = new UINT8[2 * COLOR_H * COLOR_W];
		memset(depthImages[i], 0, DEPTH_H * DEPTH_W * sizeof(UINT16));
		memset(colorImages[i], 0, 2 * COLOR_H * COLOR_W * sizeof(UINT8));
	}
	colorImagesRGB = new RGBQUAD*[MAX_CAMERAS];
	for (int i = 0; i < MAX_CAMERAS; i++) {
		colorImagesRGB[i] = new RGBQUAD[COLOR_H * COLOR_W];
	}
	depth2color = new Transformation[MAX_CAMERAS];
	color2depth = new Transformation[MAX_CAMERAS];
	depthIntrinsics = new Intrinsics[MAX_CAMERAS];
	colorIntrinsics = new Intrinsics[MAX_CAMERAS];
	transmission = NULL;

	rs2::context context;
	rs2::device_list deviceList = context.query_devices();
	for (int i = 0; i < deviceList.size(); i++) {
		enableDevice(deviceList[i]);
		std::cout << "Device " << i << " open." << std::endl;
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
	if (alignColorMap != NULL) {
		delete alignColorMap;
	}
	if (depthImages != NULL) {
		for (int i = 0; i < MAX_CAMERAS; i++) {
			if (depthImages[i] != NULL) {
				delete depthImages[i];
			}
		}
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
	if (depth2color != NULL) {
		delete[] depth2color;
	}
	if (color2depth != NULL) {
		delete[] color2depth;
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
	cfg.enable_stream(RS2_STREAM_DEPTH, DEPTH_W, DEPTH_H, RS2_FORMAT_Z16, CAMERA_FPS);
	cfg.enable_stream(RS2_STREAM_COLOR, COLOR_W, COLOR_H, RS2_FORMAT_YUYV, CAMERA_FPS);
	cfg.disable_stream(RS2_STREAM_INFRARED, 1);
	cfg.disable_stream(RS2_STREAM_INFRARED, 2);
	
	std::vector<rs2::sensor> sensors = device.query_sensors();
	for (int i = 0; i < sensors.size(); i++) {
		if (strcmp(sensors[i].get_info(RS2_CAMERA_INFO_NAME), "Stereo Module") == 0) {
			sensors[i].set_option(RS2_OPTION_ENABLE_AUTO_EXPOSURE, 0);
			float depth_unit = sensors[i].get_option(RS2_OPTION_DEPTH_UNITS);
			float stereo_baseline = sensors[i].get_option(RS2_OPTION_STEREO_BASELINE) * 0.001;
			convertFactors.push_back(stereo_baseline * (1 << 5) / depth_unit);
		}
		if (strcmp(sensors[i].get_info(RS2_CAMERA_INFO_NAME), "RGB Camera") == 0) {
			sensors[i].set_option(RS2_OPTION_ENABLE_AUTO_EXPOSURE, 0);
			sensors[i].set_option(RS2_OPTION_ENABLE_AUTO_WHITE_BALANCE, 0);
			sensors[i].set_option(RS2_OPTION_GAIN, 128);
			sensors[i].set_option(RS2_OPTION_SHARPNESS, 50);
			sensors[i].set_option(RS2_OPTION_EXPOSURE, 312);
		}
	}

	rs2::pipeline pipeline;
	pipeline.start(cfg);

	devices.push_back(pipeline);
}

int RealsenseGrabber::getRGBD(float*& depthImages_device, RGBQUAD*& colorImages_device, Transformation* world2depth, Transformation* world2color, Intrinsics*& depthIntrinsics, Intrinsics*& colorIntrinsics)
{
	depthImages_device = depthFilter->getCurrFrame_device();
	colorImages_device = colorFilter->getCurrFrame_device();
	depthIntrinsics = this->depthIntrinsics;
	colorIntrinsics = this->colorIntrinsics;
	bool check[MAX_CAMERAS] = { false };

	for (int deviceId = 0; deviceId < devices.size(); deviceId++) {
		rs2::pipeline pipeline = devices[deviceId];
		rs2::frameset frameset = pipeline.wait_for_frames();
		check[deviceId] = (frameset.size() == 2);

		if (check[deviceId] == false) {
			std::cout << deviceId << " Failed" << std::endl;
		}

		if (check[deviceId]) {
			rs2::stream_profile depthProfile;
			rs2::stream_profile colorProfile;

			for (int i = 0; i < frameset.size(); i++) {
				rs2::frame frame = frameset[i];
				rs2::stream_profile profile = frame.get_profile();
				rs2_intrinsics intrinsics = profile.as<rs2::video_stream_profile>().get_intrinsics();

				if (profile.stream_type() == RS2_STREAM_DEPTH) {
					depthProfile = profile;
					depthIntrinsics[deviceId].fx = intrinsics.fx;
					depthIntrinsics[deviceId].fy = intrinsics.fy;
					depthIntrinsics[deviceId].ppx = intrinsics.ppx;
					depthIntrinsics[deviceId].ppy = intrinsics.ppy;
					memcpy(depthImages[deviceId], frame.get_data(), DEPTH_H * DEPTH_W * sizeof(UINT16));
				}
				if (profile.stream_type() == RS2_STREAM_COLOR) {
					colorProfile = profile;
					colorIntrinsics[deviceId].fx = intrinsics.fx;
					colorIntrinsics[deviceId].fy = intrinsics.fy;
					colorIntrinsics[deviceId].ppx = intrinsics.ppx;
					colorIntrinsics[deviceId].ppy = intrinsics.ppy;
					memcpy(colorImages[deviceId], frame.get_data(), 2 * COLOR_H * COLOR_W * sizeof(UINT8));
				}
			}

			rs2_extrinsics d2cExtrinsics = depthProfile.get_extrinsics_to(colorProfile);
			depth2color[deviceId] = Transformation(d2cExtrinsics.rotation, d2cExtrinsics.translation);
			rs2_extrinsics c2dExtrinsics = colorProfile.get_extrinsics_to(depthProfile);
			color2depth[deviceId] = Transformation(c2dExtrinsics.rotation, c2dExtrinsics.translation);
		}
	}
	
	for (int i = 0; i < devices.size(); i++) {
		if (check[i]) {
			depthFilter->setConvertFactor(i, depthIntrinsics[i].fx * convertFactors[i]);
			depthFilter->process(i, depthImages[i]);
			colorFilter->process(i, colorImages[i]);
		}
	}

	colorImages_device = alignColorMap->getAlignedColor_device(devices.size(), check, depthImages_device, colorImages_device, depthIntrinsics, colorIntrinsics, depth2color);
	for (int i = 0; i < devices.size(); i++) {
		if (check[i]) {
			world2depth[i] = color2depth[i] * world2color[i];
			colorIntrinsics[i] = depthIntrinsics[i].zoom((float)COLOR_W / DEPTH_W, (float)COLOR_H / DEPTH_H);
		}
	}

	if (transmission != NULL && transmission->isConnected) {
		transmission->prepareSendFrame(devices.size(), check, depthImages_device, colorImages_device, world2depth, depthIntrinsics, colorIntrinsics);
	}

	return devices.size();
}

int RealsenseGrabber::getRGB(RGBQUAD**& colorImages, Intrinsics*& colorIntrinsics)
{
	colorImages = this->colorImagesRGB;
	colorIntrinsics = this->colorIntrinsics;
	bool check[MAX_CAMERAS];

	for (int deviceId = 0; deviceId < devices.size(); deviceId++) {
		rs2::pipeline pipeline = devices[deviceId];
		rs2::frameset frameset = pipeline.wait_for_frames();
		check[deviceId] = (frameset.size() == 2);

		if (check[deviceId]) {
			for (int i = 0; i < frameset.size(); i++) {
				rs2::frame frame = frameset[i];
				rs2::stream_profile profile = frame.get_profile();
				rs2_intrinsics intrinsics = profile.as<rs2::video_stream_profile>().get_intrinsics();

				if (profile.stream_type() == RS2_STREAM_COLOR) {
					colorIntrinsics[deviceId].fx = intrinsics.fx;
					colorIntrinsics[deviceId].fy = intrinsics.fy;
					colorIntrinsics[deviceId].ppx = intrinsics.ppx;
					colorIntrinsics[deviceId].ppy = intrinsics.ppy;
					memcpy(this->colorImages[deviceId], frame.get_data(), 2 * COLOR_W * COLOR_H * sizeof(UINT8));
				}
			}
		}
	}

	RGBQUAD* colorImages_device = colorFilter->getCurrFrame_device();
	for (int i = 0; i < devices.size(); i++) {
		if (check[i]) {
			colorFilter->process(i, this->colorImages[i]);
			cudaMemcpy(this->colorImagesRGB[i], colorImages_device + i * COLOR_W * COLOR_H, COLOR_W * COLOR_H * sizeof(RGBQUAD), cudaMemcpyDeviceToHost);
		}
	}

	return devices.size();
}

void RealsenseGrabber::saveBackground() {
	if (alignColorMap->isBackgroundOn()) {
		alignColorMap->disableBackground();
	} else {
		alignColorMap->enableBackground(depthFilter->getCurrFrame_device());
	}
	Configuration::saveBackground(alignColorMap);
}

void RealsenseGrabber::loadBackground()
{
	Configuration::loadBackground(alignColorMap);
}
