#include "RealsenseGrabber.h"
#include "Timer.h"
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
		depthImages[i] = new UINT16[(2 * DEPTH_H) * (2 * DEPTH_W)];
		colorImages[i] = new UINT8[2 * COLOR_H * COLOR_W];
		memset(depthImages[i], 0, (2 * DEPTH_H) * (2 * DEPTH_W) * sizeof(UINT16));
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

int RealsenseGrabber::getRGBD(float*& depthImages_device, RGBQUAD*& colorImages_device, Transformation*& depth2color, Transformation*& color2depth, Intrinsics*& depthIntrinsics, Intrinsics*& colorIntrinsics)
{
	for (int deviceId = 0; deviceId < devices.size(); deviceId++) {
		rs2::pipeline pipeline = devices[deviceId];
		rs2::frameset frameset = pipeline.wait_for_frames();

		if (frameset.size() > 0) {
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
					memcpy(this->depthImages[deviceId], frame.get_data(), (2 * DEPTH_H) * (2 * DEPTH_W) * sizeof(UINT16));
				}
				if (profile.stream_type() == RS2_STREAM_COLOR) {
					colorProfile = profile;
					this->colorIntrinsics[deviceId].fx = intrinsics.fx;
					this->colorIntrinsics[deviceId].fy = intrinsics.fy;
					this->colorIntrinsics[deviceId].ppx = intrinsics.ppx;
					this->colorIntrinsics[deviceId].ppy = intrinsics.ppy;
					memcpy(this->colorImages[deviceId], frame.get_data(), 2 * COLOR_H * COLOR_W * sizeof(UINT8));
				}
			}

			rs2_extrinsics d2cExtrinsics = depthProfile.get_extrinsics_to(colorProfile);
			this->depth2color[deviceId] = Transformation(d2cExtrinsics.rotation, d2cExtrinsics.translation);
			rs2_extrinsics c2dExtrinsics = colorProfile.get_extrinsics_to(depthProfile);
			this->color2depth[deviceId] = Transformation(c2dExtrinsics.rotation, c2dExtrinsics.translation);
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
	depth2color = this->depth2color;
	color2depth = this->color2depth;
	depthIntrinsics = this->depthIntrinsics;
	colorIntrinsics = this->colorIntrinsics;

	RGBQUAD* alignedColorMap = alignColorMap->getAlignedColor_device(devices.size(), depthImages_device, colorImages_device, depthIntrinsics, colorIntrinsics, depth2color);
	colorImages_device = alignedColorMap;
	for (int i = 0; i < devices.size(); i++) {
		colorIntrinsics[i] = depthIntrinsics[i].zoom((float)COLOR_W / DEPTH_W, (float)COLOR_H / DEPTH_H);
		depth2color[i] = Transformation();
		color2depth[i] = Transformation();
	}

	return devices.size();
}

int RealsenseGrabber::getRGB(RGBQUAD**& colorImages, Intrinsics*& colorIntrinsics)
{
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
