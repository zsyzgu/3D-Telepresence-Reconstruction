#include "RealsenseGrabber.h"
#include "Timer.h"
#include "librealsense2/hpp/rs_sensor.hpp"
#include "librealsense2/hpp/rs_processing.hpp"

RealsenseGrabber::RealsenseGrabber()
{
	depthFilter = new DepthFilter();
	for (int i = 0; i < MAX_CAMERAS; i++) {
		decimationFilter[i] = new rs2::decimation_filter();
	}
	depthImages = new UINT16*[MAX_CAMERAS];
	colorImages = new RGBQUAD*[MAX_CAMERAS];
	for (int i = 0; i < MAX_CAMERAS; i++) {
		depthImages[i] = NULL;
		colorImages[i] = NULL;
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
	for (int i = 0; i < devices.size(); i++) {
		if (depthImages[i] != NULL) {
			cudaHostUnregister(depthImages[i]);
		}
		if (colorImages[i] != NULL) {
			cudaHostUnregister(colorImages[i]);
		}
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
		delete[] colorImages;
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
	UINT8 U, V;
	for (int i = 0; i < COLOR_H * COLOR_W; i++) {
		if ((i & 1) == 0) {
			U = src[i * 2 + 1];
			V = src[i * 2 + 3];
		}
		UINT8 Y = src[i * 2];

		dst[i].rgbRed = Y;
		dst[i].rgbGreen = Y;
		dst[i].rgbBlue = Y;
	}
}

int RealsenseGrabber::getRGBD(float*& depthImages_device, RGBQUAD**& colorImages, Transformation*& depthTrans, Intrinsics*& depthIntrinsics, Intrinsics*& colorIntrinsics)
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
					frame = decimationFilter[deviceId]->process(frame);

					if (this->depthImages[deviceId] == NULL) {
						this->depthImages[deviceId] = (UINT16*)frame.get_data();
						cudaHostRegister(this->depthImages[deviceId], DEPTH_H * DEPTH_W * sizeof(UINT16), cudaHostRegisterPortable);
						this->depthIntrinsics[deviceId].fx = intrinsics.fx * 0.5;
						this->depthIntrinsics[deviceId].fy = intrinsics.fy * 0.5;
						this->depthIntrinsics[deviceId].ppx = intrinsics.ppx * 0.5;
						this->depthIntrinsics[deviceId].ppy = intrinsics.ppy * 0.5;
						depthFilter->setConvertFactor(deviceId, intrinsics.fx * 0.5 * convertFactors[deviceId]);
					}
				}
				if (profile.stream_type() == RS2_STREAM_COLOR) {
					colorProfile = profile;
					if (this->colorImages[deviceId] == NULL) {
						this->colorImages[deviceId] = new RGBQUAD[COLOR_H * COLOR_W];
						cudaHostRegister(this->colorImages[deviceId], COLOR_H * COLOR_W * sizeof(uchar4), cudaHostRegisterPortable);
						this->colorIntrinsics[deviceId].fx = intrinsics.fx;
						this->colorIntrinsics[deviceId].fy = intrinsics.fy;
						this->colorIntrinsics[deviceId].ppx = intrinsics.ppx;
						this->colorIntrinsics[deviceId].ppy = intrinsics.ppy;
					}

					convertYUVtoRGBA((UINT8*)frame.get_data(), this->colorImages[deviceId]);
				}
			}

			rs2_extrinsics extrinsics = colorProfile.get_extrinsics_to(depthProfile);
			this->depthTrans[deviceId] = Transformation(extrinsics.rotation, extrinsics.translation);
		}
	}

	for (int deviceId = 0; deviceId < devices.size(); deviceId++) {
		if (this->depthImages[deviceId] != NULL) {
			depthFilter->process(deviceId, this->depthImages[deviceId]);
		}
	}

	colorImages = this->colorImages;
	depthImages_device = depthFilter->getCurrFrame_device();
	depthTrans = this->depthTrans;
	depthIntrinsics = this->depthIntrinsics;
	colorIntrinsics = this->colorIntrinsics;

	return devices.size();
}

int RealsenseGrabber::getRGB(RGBQUAD**& colorImages, Intrinsics*& colorIntrinsics)
{
	float* depthImages_device;
	Transformation* depthTrans;
	Intrinsics* depthIntrinsics;
	return getRGBD(depthImages_device, colorImages, depthTrans, depthIntrinsics, colorIntrinsics);
}
