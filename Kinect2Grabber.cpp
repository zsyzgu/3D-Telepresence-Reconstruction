#include "Kinect2Grabber.h"
#include "Timer.h"
#include <pcl/filters/fast_bilateral_omp.h>
#include <omp.h>

extern "C"
void cudaBilateralFiltering(UINT16* depthData, float* depth);

namespace pcl
{
	pcl::Kinect2Grabber::Kinect2Grabber()
		: sensor(nullptr)
		, mapper(nullptr)
		, colorSource(nullptr)
		, colorReader(nullptr)
		, depthSource(nullptr)
		, depthReader(nullptr)
		, result(S_OK)
		, colorWidth(1920)
		, colorHeight(1080)
		, colorBuffer()
		, W(512)
		, H(424)
		, depthBuffer()
	{
		// Create Sensor Instance
		result = GetDefaultKinectSensor(&sensor);
		if (FAILED(result)) {
			throw std::exception("Exception : GetDefaultKinectSensor()");
		}

		// Open Sensor
		result = sensor->Open();
		if (FAILED(result)) {
			throw std::exception("Exception : IKinectSensor::Open()");
		}

		// Retrieved Coordinate Mapper
		result = sensor->get_CoordinateMapper(&mapper);
		if (FAILED(result)) {
			throw std::exception("Exception : IKinectSensor::get_CoordinateMapper()");
		}

		// Retrieved Color Frame Source
		result = sensor->get_ColorFrameSource(&colorSource);
		if (FAILED(result)) {
			throw std::exception("Exception : IKinectSensor::get_ColorFrameSource()");
		}

		// Retrieved Depth Frame Source
		result = sensor->get_DepthFrameSource(&depthSource);
		if (FAILED(result)) {
			throw std::exception("Exception : IKinectSensor::get_DepthFrameSource()");
		}

		// Retrieved Color Frame Size
		IFrameDescription* colorDescription;
		result = colorSource->get_FrameDescription(&colorDescription);
		if (FAILED(result)) {
			throw std::exception("Exception : IColorFrameSource::get_FrameDescription()");
		}

		result = colorDescription->get_Width(&colorWidth); // 1920
		if (FAILED(result)) {
			throw std::exception("Exception : IFrameDescription::get_Width()");
		}

		result = colorDescription->get_Height(&colorHeight); // 1080
		if (FAILED(result)) {
			throw std::exception("Exception : IFrameDescription::get_Height()");
		}

		SafeRelease(colorDescription);

		// To Reserve Color Frame Buffer
		colorBuffer.resize(colorWidth * colorHeight);

		// Retrieved Depth Frame Size
		IFrameDescription* depthDescription;
		result = depthSource->get_FrameDescription(&depthDescription);
		if (FAILED(result)) {
			throw std::exception("Exception : IDepthFrameSource::get_FrameDescription()");
		}

		result = depthDescription->get_Width(&W); // 512
		if (FAILED(result)) {
			throw std::exception("Exception : IFrameDescription::get_Width()");
		}

		result = depthDescription->get_Height(&H); // 424
		if (FAILED(result)) {
			throw std::exception("Exception : IFrameDescription::get_Height()");
		}

		SafeRelease(depthDescription);

		// To Reserve Depth Frame Buffer
		depthBuffer.resize(W * H);

		start();
	}

	pcl::Kinect2Grabber::~Kinect2Grabber() throw()
	{
		if (sensor) {
			sensor->Close();
		}
		SafeRelease(sensor);
		SafeRelease(mapper);
		SafeRelease(colorSource);
		SafeRelease(colorReader);
		SafeRelease(depthSource);
		SafeRelease(depthReader);
	}

	void pcl::Kinect2Grabber::start()
	{
		// Open Color Frame Reader
		result = colorSource->OpenReader(&colorReader);
		if (FAILED(result)) {
			throw std::exception("Exception : IColorFrameSource::OpenReader()");
		}

		// Open Depth Frame Reader
		result = depthSource->OpenReader(&depthReader);
		if (FAILED(result)) {
			throw std::exception("Exception : IDepthFrameSource::OpenReader()");
		}
	}

	pcl::PointCloud<pcl::PointXYZRGB>::Ptr Kinect2Grabber::getPointCloud()
	{
		IDepthFrame* depthFrame = nullptr;
		depthReader->AcquireLatestFrame(&depthFrame);
		depthFrame->CopyFrameDataToArray(depthBuffer.size(), &depthBuffer[0]);
		SafeRelease(depthFrame);

		IColorFrame* colorFrame = nullptr;
		colorReader->AcquireLatestFrame(&colorFrame);
		colorFrame->CopyConvertedFrameDataToArray(colorBuffer.size() * sizeof(RGBQUAD), reinterpret_cast<BYTE*>(&colorBuffer[0]), ColorImageFormat::ColorImageFormat_Bgra);
		SafeRelease(colorFrame);

		return convertRGBDepthToPointXYZRGB(&colorBuffer[0], &depthBuffer[0]);
	}

	pcl::PointCloud<pcl::PointXYZRGB>::Ptr pcl::Kinect2Grabber::convertRGBDepthToPointXYZRGB(RGBQUAD* colorData, UINT16* depthData)
	{
		spatialFiltering(depthData);
		temporalFiltering(depthData);
		float* depthFloat = new float[H * W];
		bilateralFiltering(depthData, depthFloat);

		pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>());

		cloud->width = static_cast<uint32_t>(W);
		cloud->height = static_cast<uint32_t>(H);
		cloud->is_dense = false;

		cloud->points.resize(H * W);

		UINT32 tableCount;
		PointF* depthToCameraSpaceTable = nullptr;
		mapper->GetDepthFrameToCameraSpaceTable(&tableCount, &depthToCameraSpaceTable);

		ColorSpacePoint* depthToColorSpaceTable = new ColorSpacePoint[W * H];
		mapper->MapDepthFrameToColorSpace(W * H, depthData, W * H, depthToColorSpaceTable);

#pragma omp parallel for schedule(dynamic, 1)
		for (int y = 0; y < H; y++) {
			int id = y * W;
			pcl::PointXYZRGB* pt = &cloud->points[id];
			for (int x = 0; x < W; x++, pt++, id++) {
				DepthSpacePoint depthSpacePoint = { static_cast<float>(x), static_cast<float>(y) };
				float depth = depthFloat[id];
				if (depth != 0) {
					ColorSpacePoint colorSpacePoint = depthToColorSpaceTable[id];
					int colorX = static_cast<int>(std::floor(colorSpacePoint.X));
					int colorY = static_cast<int>(std::floor(colorSpacePoint.Y));
					if ((0 <= colorX) && (colorX + 1 < colorWidth) && (0 <= colorY) && (colorY + 1 < colorHeight)) {
						// Coordinate Mapping Depth to Color Space, and Setting PointCloud RGB
						RGBQUAD colorLU = colorData[colorY * colorWidth + colorX];
						RGBQUAD colorRU = colorData[colorY * colorWidth + (colorX + 1)];
						RGBQUAD colorLB = colorData[(colorY + 1) * colorWidth + colorX];
						RGBQUAD colorRB = colorData[(colorY + 1) * colorWidth + (colorX + 1)];
						float u = colorSpacePoint.X - colorX;
						float v = colorSpacePoint.Y - colorY;
						pt->b = colorLU.rgbBlue * (1 - u) * (1 - v) + colorRU.rgbBlue * u * (1 - v) + colorLB.rgbBlue * (1 - u) * v + colorRB.rgbBlue * u * v;
						pt->g = colorLU.rgbGreen * (1 - u) * (1 - v) + colorRU.rgbGreen * u * (1 - v) + colorLB.rgbGreen * (1 - u) * v + colorRB.rgbGreen * u * v;
						pt->r = colorLU.rgbRed * (1 - u) * (1 - v) + colorRU.rgbRed * u * (1 - v) + colorLB.rgbRed * (1 - u) * v + colorRB.rgbRed * u * v;
						// Coordinate Mapping Depth to Camera Space, and Setting PointCloud XYZs
						PointF spacePoint = depthToCameraSpaceTable[id];
						pt->x = spacePoint.X * depth * 0.001f;
						pt->y = spacePoint.Y * depth * 0.001f;
						pt->z = depth * 0.001f;
					}
				}
			}
		}

		delete[] depthToCameraSpaceTable;
		delete[] depthToColorSpaceTable;
		delete[] depthFloat;
		
		return cloud;
	}

	void pcl::Kinect2Grabber::spatialFiltering(UINT16* depth) {
		static const int n = 512 * 424;
		static UINT16 rawDepth[n];

		memcpy(rawDepth, depth, n * sizeof(UINT16));
		
#pragma omp parallel for schedule(dynamic, 1)
		for (int y = 0; y < H; y++) {
			for (int x = 0; x < W; x++) {
				int index = y * W + x;
				if (rawDepth[index] == 0) {
					int num = 0;
					int sum = 0;
					for (int dx = -2; dx <= 2; dx++) {
						for (int dy = -2; dy <= 2; dy++) {
							if (dx != 0 || dy != 0) {
								int xSearch = x + dx;
								int ySearch = y + dy;

								if (0 <= xSearch && xSearch < W && 0 <= ySearch && ySearch < H) {
									int searchIndex = ySearch * W + xSearch;
									if (rawDepth[searchIndex] != 0) {
										num++;
										sum += rawDepth[searchIndex];
									}
								}
							}
						}
					}
					if (num != 0) {
						depth[index] = (sum + (num >> 1)) / num;
					}
				}
			}
		}
	}

	void pcl::Kinect2Grabber::temporalFiltering(UINT16* depth) {
		// Abstract: The system error of a pixel from the depth image is nearly white noise. We use several frames to smooth it.
		// Implementation: If a pixel not change a lot in this frame, we would smooth it by averaging the pixels in several frames.

		static const int QUEUE_LENGTH = 5;
		static const int THRESHOLD = 10;
		static const int n = 512 * 424;
		static UINT16 depthQueue[QUEUE_LENGTH][n];
		static int t = 0;

		memcpy(depthQueue[t], depth, n * sizeof(UINT16));

#pragma omp parallel for schedule(dynamic, 1)
		for (int y = 0; y < H; y++) {
			for (int x = 0; x < W; x++) {
				int index = y * W + x;
				int sum = 0;
				int num = 0;
				for (int i = 0; i < QUEUE_LENGTH; i++) {
					if (depthQueue[i][index] != 0) {
						sum += depthQueue[i][index];
						num++;
					}
				}
				if (num == 0) {
					depth[index] = 0;
				}
				else {
					depth[index] = (UINT16)(sum / num);
					if (abs(depthQueue[t][index] - depth[index]) > THRESHOLD) {
						depth[index] = depthQueue[t][index];
						for (int i = 0; i < QUEUE_LENGTH; i++) {
							depthQueue[t][index] = 0;
						}
					}
				}
			}
		}

		t = (t + 1) % QUEUE_LENGTH;
	}

	void Kinect2Grabber::bilateralFiltering(UINT16* depthData, float* depth)
	{
		cudaBilateralFiltering(depthData, depth);
#pragma omp parallel for
		for (int i = 0; i < H * W; i++) {
			depthData[i] = (UINT16)depth[i];
		}
	}
}

/*
Another implementation of spatialFiltering()

// Abstract: Depth image captured by Kinect2 is usually missing some pixels. We use their neighbor pixels to estimate them.
// Implemention [9 * O(n)]: Constantly find the unkonwn pixel with the most known neighbours, and assign it by averaging its neighbours.

static const int n = 512 * 424;
static int pixelQueue[9][n];
int tot[9] = {0};
int cnt[9] = {0};
byte* uncertainty = new byte[n];

for (int y = 0; y < H; y++) {
for (int x = 0; x < W; x++) {
int index = y * W + x;
if (depthData[index] == 0) {
int lv = 0;
for (int dx = -1; dx <= 1; dx++) {
for (int dy = -1; dy <= 1; dy++) {
if (dx != 0 || dy != 0) {
int xSearch = x + dx;
int ySearch = y + dy;

if (0 <= xSearch && xSearch < W && 0 <= ySearch && ySearch < H) {
int searchIndex = ySearch * W + xSearch;
if (depthData[searchIndex] == 0) {
lv++;
}
} else {
lv++;
}
}
}
}
uncertainty[index] = lv;
pixelQueue[lv][tot[lv]++] = index;
}
}
}

for (int lv = 0; lv < 9; ) {
if (cnt[lv] >= tot[lv]) {
lv++;
continue;
}

int index = pixelQueue[lv][cnt[lv]++];
if (uncertainty[index] != lv) {
continue;
}

int y = index / W;
int x = index % W;
int sum = 0;
int num = 0;
for (int dx = -1; dx <= 1; dx++) {
for (int dy = -1; dy <= 1; dy++) {
if (dx != 0 || dy != 0) {
int xSearch = x + dx;
int ySearch = y + dy;

if (0 <= xSearch && xSearch < W && 0 <= ySearch && ySearch < H) {
int searchIndex = ySearch * W + xSearch;
if (depthData[searchIndex] == 0) {
int searchLv = --uncertainty[searchIndex];
pixelQueue[searchLv][tot[searchLv]++] = searchIndex;
} else {
sum += depthData[searchIndex];
num++;
}
}
}
}
}

if (num == 0) {
return; //All pixels are zero
} else {
depthData[index] = (UINT16)(sum / num);
}

if (lv - 1 >= 0 && cnt[lv - 1] < tot[lv - 1]) {
lv--;
}
}

delete[] uncertainty;*/