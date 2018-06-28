#include "RealsenseGrabber.h"

RealsenseGrabber::RealsenseGrabber()
{
	for (int i = 0; i < MAX_CAMERAS; i++) {
		decimationFilter[i] = new rs2::decimation_filter();
		spatialFilter[i] = new rs2::spatial_filter();
		temporalFilter[i] = new rs2::temporal_filter();
		toDisparityFilter[i] = new rs2::disparity_transform(true);
		toDepthFilter[i] = new rs2::disparity_transform(false);
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
		if (spatialFilter[i] != NULL) {
			delete spatialFilter[i];
		}
		if (temporalFilter[i] != NULL) {
			delete temporalFilter[i];
		}
		if (toDisparityFilter[i] != NULL) {
			delete toDisparityFilter[i];
		}
		if (toDepthFilter[i] != NULL) {
			delete toDepthFilter[i];
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

	if (device.get_info(RS2_CAMERA_INFO_NAME) == "Platform Camera") {
		return;
	}

	rs2::config cfg;
	cfg.enable_device(serialNumber);
	cfg.enable_stream(RS2_STREAM_DEPTH, DEPTH_W * 2, DEPTH_H * 2, RS2_FORMAT_Z16, 60);
	cfg.enable_stream(RS2_STREAM_COLOR, COLOR_W, COLOR_H, RS2_FORMAT_RGBA8, 60);

	rs2::pipeline pipeline;
	pipeline.start(cfg);

	devices.push_back(pipeline);
}

int RealsenseGrabber::getRGBD(UINT16**& depthImages, RGBQUAD**& colorImages, Transformation*& depthTrans, Intrinsics*& depthIntrinsics, Intrinsics*& colorIntrinsics)
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
					frame = decimationFilter[deviceId]->process(frame); //Decimation Filter: pixel *= 0.5
					frame = toDisparityFilter[deviceId]->process(frame);
					frame = spatialFilter[deviceId]->process(frame);
					frame = temporalFilter[deviceId]->process(frame);
					frame = toDepthFilter[deviceId]->process(frame);
					if (this->depthImages[deviceId] == NULL) {
						this->depthImages[deviceId] = (UINT16*)frame.get_data();
						cudaHostRegister(this->depthImages[deviceId], DEPTH_H * DEPTH_W * sizeof(UINT16), cudaHostRegisterPortable);
						this->depthIntrinsics[deviceId].fx = intrinsics.fx * 0.5;
						this->depthIntrinsics[deviceId].fy = intrinsics.fy * 0.5;
						this->depthIntrinsics[deviceId].ppx = intrinsics.ppx * 0.5;
						this->depthIntrinsics[deviceId].ppy = intrinsics.ppy * 0.5;
					}
				}
				if (profile.stream_type() == RS2_STREAM_COLOR) {
					colorProfile = profile;
					if (this->colorImages[deviceId] == NULL) {
						this->colorImages[deviceId] = (RGBQUAD*)frame.get_data();
						cudaHostRegister(this->colorImages[deviceId], COLOR_H * COLOR_W * sizeof(uchar4), cudaHostRegisterPortable);
						this->colorIntrinsics[deviceId].fx = intrinsics.fx;
						this->colorIntrinsics[deviceId].fy = intrinsics.fy;
						this->colorIntrinsics[deviceId].ppx = intrinsics.ppx;
						this->colorIntrinsics[deviceId].ppy = intrinsics.ppy;
					}
				}
			}

			rs2_extrinsics extrinsics = colorProfile.get_extrinsics_to(depthProfile);
			this->depthTrans[deviceId] = Transformation(extrinsics.rotation, extrinsics.translation);
		}
	}
	depthImages = this->depthImages;
	colorImages = this->colorImages;
	depthTrans = this->depthTrans;
	depthIntrinsics = this->depthIntrinsics;
	colorIntrinsics = this->colorIntrinsics;
	return devices.size();
}
