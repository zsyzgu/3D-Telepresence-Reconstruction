#ifndef KINECT_2_GRABBER
#define KINECT_2_GRABBER

#define NOMINMAX

#include <Windows.h>
#include <Kinect.h>

#include <pcl/io/boost.h>
#include <pcl/io/grabber.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

namespace pcl
{
	template <typename T> class pcl::PointCloud;

	template<class Interface>
	inline void SafeRelease(Interface *& IRelease)
	{
		if (IRelease != NULL) {
			IRelease->Release();
			IRelease = NULL;
		}
	}

	class Kinect2Grabber
	{
	public:
		Kinect2Grabber();
		~Kinect2Grabber() throw ();

		pcl::PointCloud<pcl::PointXYZRGB>::Ptr getPointCloud();
		void updateBackground();
		void updateDepthAndColor();
		void getDepthAndColor(UINT16*& depthData, RGBQUAD*& colorData);

	protected:
		void loadBackground();
		void calnForegroundMask();
		pcl::PointCloud<pcl::PointXYZRGB>::Ptr convertRGBDepthToPointXYZRGB();
		void spatialFiltering();
		void temporalFiltering();
		void bilateralFiltering();

		HRESULT result;
		IKinectSensor* sensor;
		ICoordinateMapper* mapper;
		IColorFrameSource* colorSource;
		IColorFrameReader* colorReader;
		IDepthFrameSource* depthSource;
		IDepthFrameReader* depthReader;

		int colorWidth;
		int colorHeight;
		std::vector<RGBQUAD> colorBuffer;
		std::vector<UINT16> depthBuffer;

		int W; //Width & Height of depth image
		int H;
		UINT16* backgroundDepth;
		RGBQUAD* backgroundColor;
		bool* foregroundMask;
		UINT16* depthData;
		RGBQUAD* colorData;
	};
}

#endif KINECT_2_GRABBER
