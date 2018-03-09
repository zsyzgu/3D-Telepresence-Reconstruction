#include "Kinect2Grabber.h"
#include "Timer.h"
#include <pcl/filters/fast_bilateral_omp.h>
#include <omp.h>
#include<sstream>
#include<string>

extern "C"
void cudaBilateralFiltering(UINT16* depthData);

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

		backgroundDepth = new UINT16[H * W];
		backgroundColor = new RGBQUAD[H * W];
		foregroundMask = new bool[H * W];
		loadBackground();
		depthData = &depthBuffer[0];
		colorData = new RGBQUAD[H * W];
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
		delete[] backgroundDepth;
		delete[] backgroundColor;
		delete[] foregroundMask;
		delete[] colorData;
	}

	void pcl::Kinect2Grabber::loadBackground() {
		FILE* fin = fopen("background", "r");
		if (fin != NULL) {
			int i = 0;
			while (!feof(fin) && i < H * W) {
				fscanf(fin, "%hd %hd %hd %hd", &backgroundDepth[i], &backgroundColor[i].rgbRed, &backgroundColor[i].rgbGreen, &backgroundColor[i].rgbBlue);
				i++;
			}
		}
	}

	void pcl::Kinect2Grabber::updateBackground() {
		FILE* fout = fopen("background", "w");
		if (fout != NULL) {
			for (int i = 0; i < H * W; i++) {
				backgroundDepth[i] = depthData[i];
				backgroundColor[i] = colorData[i];
				fprintf(fout, "%hd %hd %hd %hd\n", backgroundDepth[i], backgroundColor[i].rgbRed, backgroundColor[i].rgbGreen, backgroundColor[i].rgbBlue);
			}
		}
	}

	void pcl::Kinect2Grabber::calnForegroundMask() {
#pragma omp parallel for schedule(static, 500)
		for (int i = 0; i < H * W; i++) {
			if (backgroundDepth[i] == 0) {
				foregroundMask[i] = true;
			}
			else {
				int depthDiff = abs(depthData[i] - backgroundDepth[i]);
				int colorDiff = abs(colorData[i].rgbRed - backgroundColor[i].rgbRed) + abs(colorData[i].rgbGreen - backgroundColor[i].rgbGreen) + abs(colorData[i].rgbBlue - backgroundColor[i].rgbBlue);
				if (depthDiff + 5.0 * colorDiff / (3 * 255) >= 10) {
					foregroundMask[i] = true;
				}
				else {
					foregroundMask[i] = false;
				}
			}
		}
	}

	pcl::PointCloud<pcl::PointXYZRGB>::Ptr Kinect2Grabber::getPointCloud()
	{
		updateDepthAndColor();
		return convertRGBDepthToPointXYZRGB();
	}


	void Kinect2Grabber::updateDepthAndColor()
	{
		IDepthFrame* depthFrame = nullptr;
		depthReader->AcquireLatestFrame(&depthFrame);
		if (depthFrame != NULL) {
			depthFrame->CopyFrameDataToArray(depthBuffer.size(), &depthBuffer[0]);
			depthFrame->Release();
		}

		IColorFrame* colorFrame = nullptr;
		colorReader->AcquireLatestFrame(&colorFrame);
		if (colorFrame != NULL) {
			colorFrame->CopyConvertedFrameDataToArray(colorBuffer.size() * sizeof(RGBQUAD), reinterpret_cast<BYTE*>(&colorBuffer[0]), ColorImageFormat::ColorImageFormat_Bgra);
			colorFrame->Release();
		}

		spatialFiltering();
		temporalFiltering();
		bilateralFiltering();
		calnForegroundMask();

		ColorSpacePoint* depthToColorSpaceTable = new ColorSpacePoint[W * H];
		mapper->MapDepthFrameToColorSpace(W * H, depthData, W * H, depthToColorSpaceTable);

#pragma omp parallel for schedule(dynamic, 1)
		for (int y = 0; y < H; y++) {
			for (int x = 0; x < W; x++) {
				int id = y * W + x;
				ColorSpacePoint colorSpacePoint = depthToColorSpaceTable[id];
				int colorX = static_cast<int>(std::floor(colorSpacePoint.X));
				int colorY = static_cast<int>(std::floor(colorSpacePoint.Y));
				if (foregroundMask[id] && depthData[id] != 0 && (0 <= colorX) && (colorX + 1 < colorWidth) && (0 <= colorY) && (colorY + 1 < colorHeight)) {
					RGBQUAD colorLU = colorBuffer[colorY * colorWidth + colorX];
					RGBQUAD colorRU = colorBuffer[colorY * colorWidth + (colorX + 1)];
					RGBQUAD colorLB = colorBuffer[(colorY + 1) * colorWidth + colorX];
					RGBQUAD colorRB = colorBuffer[(colorY + 1) * colorWidth + (colorX + 1)];
					float u = colorSpacePoint.X - colorX;
					float v = colorSpacePoint.Y - colorY;
					colorData[id].rgbBlue = colorLU.rgbBlue * (1 - u) * (1 - v) + colorRU.rgbBlue * u * (1 - v) + colorLB.rgbBlue * (1 - u) * v + colorRB.rgbBlue * u * v;
					colorData[id].rgbGreen = colorLU.rgbGreen * (1 - u) * (1 - v) + colorRU.rgbGreen * u * (1 - v) + colorLB.rgbGreen * (1 - u) * v + colorRB.rgbGreen * u * v;
					colorData[id].rgbRed = colorLU.rgbRed * (1 - u) * (1 - v) + colorRU.rgbRed * u * (1 - v) + colorLB.rgbRed * (1 - u) * v + colorRB.rgbRed * u * v;
				}
				else {
					colorData[id].rgbBlue = 0;
					colorData[id].rgbGreen = 0;
					colorData[id].rgbRed = 0;
					depthData[id] = 0;
					foregroundMask[id] = false;
				}
			}
		}

		delete[] depthToColorSpaceTable;
	}

	pcl::PointCloud<pcl::PointXYZRGB>::Ptr pcl::Kinect2Grabber::convertRGBDepthToPointXYZRGB()
	{
		pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>());
		cloud->points.resize(H * W);
		cloud->width = W;
		cloud->height = H;
		cloud->is_dense = false;

		UINT32 tableCount;
		PointF* depthToCameraSpaceTable = nullptr;
		mapper->GetDepthFrameToCameraSpaceTable(&tableCount, &depthToCameraSpaceTable);

#pragma omp parallel for schedule(dynamic, 1)
		for (int y = 0; y < H; y++) {
			int id = y * W;
			pcl::PointXYZRGB* pt = &cloud->points[id];
			for (int x = 0; x < W; x++, pt++, id++) {
				if (foregroundMask[id]) {
					float depth = depthData[id];
					pt->b = colorData[id].rgbBlue;
					pt->g = colorData[id].rgbGreen;
					pt->r = colorData[id].rgbRed;
					PointF spacePoint = depthToCameraSpaceTable[id];
					pt->x = spacePoint.X * depth * 0.001f;
					pt->y = spacePoint.Y * depth * 0.001f;
					pt->z = depth * 0.001f;
				}
			}
		}

		delete[] depthToCameraSpaceTable;
		
		return cloud;
	}

	void pcl::Kinect2Grabber::spatialFiltering() {
		static const int n = 512 * 424;
		static UINT16 rawDepth[n];

		memcpy(rawDepth, depthData, n * sizeof(UINT16));
		
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
						depthData[index] = (sum + (num >> 1)) / num;
					}
				}
			}
		}
	}

	void pcl::Kinect2Grabber::temporalFiltering() {
		// Abstract: The system error of a pixel from the depth image is nearly white noise. We use several frames to smooth it.
		// Implementation: If a pixel not change a lot in this frame, we would smooth it by averaging the pixels in several frames.

		static const int QUEUE_LENGTH = 5;
		static const int THRESHOLD = 10;
		static const int n = 512 * 424;
		static UINT16 depthQueue[QUEUE_LENGTH][n];
		static int t = 0;

		memcpy(depthQueue[t], depthData, n * sizeof(UINT16));

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
					depthData[index] = 0;
				}
				else {
					depthData[index] = (UINT16)(sum / num);
					if (abs(depthQueue[t][index] - depthData[index]) > THRESHOLD) {
						depthData[index] = depthQueue[t][index];
						for (int i = 0; i < QUEUE_LENGTH; i++) {
							depthQueue[t][index] = 0;
						}
					}
				}
			}
		}

		t = (t + 1) % QUEUE_LENGTH;
	}

	void Kinect2Grabber::bilateralFiltering()
	{
		cudaBilateralFiltering(depthData);
	}
}
