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

	rs2::context context;
	for (auto&& device : context.query_devices()) {
		enableDevice(device);
	}
}

RealsenseGrabber::~RealsenseGrabber()
{
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
		delete depthImages;
	}
	if (colorImages != NULL) {
		delete colorImages;
	}
}


void RealsenseGrabber::enableDevice(rs2::device device)
{
	std::string serialNumber(device.get_info(RS2_CAMERA_INFO_SERIAL_NUMBER));
	std::lock_guard<std::mutex> lock(_mutex);

	if (devices.find(serialNumber) != devices.end()) {
		return;
	}
	if (device.get_info(RS2_CAMERA_INFO_NAME) == "Platform Camera") {
		return;
	}

	rs2::config cfg;
	cfg.enable_device(serialNumber);
	cfg.enable_stream(RS2_STREAM_DEPTH, DEPTH_W * 2, DEPTH_H * 2, RS2_FORMAT_Z16, 60);
	cfg.enable_stream(RS2_STREAM_COLOR, COLOR_W, COLOR_H, RS2_FORMAT_RGBA8, 60);

	rs2::pipeline pipeline;
	pipeline.start(cfg);

	devices[serialNumber] = pipeline;
}

int RealsenseGrabber::getRGBD(UINT16**& depthImages, RGBQUAD**& colorImages)
{
	std::lock_guard<std::mutex> lock(_mutex);
	int deviceId = 0;
	for (auto&& view : devices) {
		rs2::frameset frameset;
		if (view.second.poll_for_frames(&frameset) && frameset.size() > 0) {
			for (int i = 0; i < frameset.size(); i++) {
				rs2::frame frame = frameset[i];
				rs2::stream_profile profile = frame.get_profile();

				if (profile.stream_type() == RS2_STREAM_DEPTH) {
					frame = decimationFilter[deviceId]->process(frame);
					frame = toDisparityFilter[deviceId]->process(frame);
					frame = spatialFilter[deviceId]->process(frame);
					frame = temporalFilter[deviceId]->process(frame);
					frame = toDepthFilter[deviceId]->process(frame);
					this->depthImages[deviceId] = (UINT16*)frame.get_data();
				}
				if (profile.stream_type() == RS2_STREAM_COLOR) {
					this->colorImages[deviceId] = (RGBQUAD*)frame.get_data();
				}
			}
			deviceId++;
		}
	}

	depthImages = this->depthImages;
	colorImages = this->colorImages;
	return deviceId;
}
