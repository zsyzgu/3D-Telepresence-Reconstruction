#include "Timer.h"
#include "Recognition.h"
#include "Kinect2Pcd.h"
#include "PointCloudProcess.h"
#include "GpuDrawCloud.cuh"

// ===== Recognize Model from Scene =====
void recognizeModelFromScene(char* modelFileName, char* sceneFileName);

// ===== Capture Model and Scene by Kinect =====
void captureModelAndSceneByKinect(char* modelFileName, char* sceneFileName);
int capturePointCnt = 0;
Eigen::Vector2f captureWindowMin;
Eigen::Vector2f captureWindowMax;
void mouseEventOccurred(const pcl::visualization::MouseEvent &event, void* viewerVoid);

// ===== Merge Two Point Cloud =====
void merge2PointClouds(char* model1FileName, char* model2FileName);




// ===== Ty CUDA =====
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

__global__ void addKernel(int *c, const int *a, const int *b)
{
	int i = threadIdx.x;
	c[i] = a[i] + b[i];
}


// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
	int *dev_a = 0;
	int *dev_b = 0;
	int *dev_c = 0;
	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	// Launch a kernel on the GPU with one thread for each element.
	addKernel << <1, size > >>(dev_c, dev_a, dev_b);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

Error:
	cudaFree(dev_c);
	cudaFree(dev_a);
	cudaFree(dev_b);

	return cudaStatus;
}





int main(int argc, char *argv[]) {
	const int arraySize = 5;
	const int a[arraySize] = { 1, 2, 3, 4, 5 };
	const int b[arraySize] = { 10, 20, 30, 40, 50 };
	int c[arraySize] = { 0 };

	cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addWithCuda failed!");
		return 1;
	}

	printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
		c[0], c[1], c[2], c[3], c[4]);

	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}
	return 0;

	//recognizeModelFromScene("chair.pcd", "scene.pcd");
	//captureModelAndSceneByKinect("model.pcd", "scene.pcd");
	//merge2PointClouds("model1.pcd", "model2.pcd");

	/*pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>());
	if (pcl::io::loadPCDFile("model6.pcd", *cloud) < 0) {
		return -1;
	}

	PointCloudProcess::mlsFiltering(cloud);

	pcl::NormalEstimation<pcl::PointXYZRGB, pcl::Normal> normalEstimation;
	pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
	pcl::search::KdTree<pcl::PointXYZRGB>::Ptr kdTree(new pcl::search::KdTree<pcl::PointXYZRGB>);
	kdTree->setInputCloud(cloud);
	normalEstimation.setInputCloud(cloud);
	normalEstimation.setSearchMethod(kdTree);
	normalEstimation.setKSearch(20);
	normalEstimation.compute(*normals);

	pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloudWithNormals(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
	pcl::concatenateFields(*cloud, *normals, *cloudWithNormals);

	pcl::io::savePCDFileASCII("model6_with_normal.pcd", *cloudWithNormals);
	return 0;*/
}

void recognizeModelFromScene(char* modelFileName, char* sceneFileName) {
	Recognition recognition;
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr model(new pcl::PointCloud<pcl::PointXYZRGB>());
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr scene(new pcl::PointCloud<pcl::PointXYZRGB>());
	if (pcl::io::loadPCDFile(modelFileName, *model) < 0) {
		std::cout << "Error loading model cloud" << std::endl;
		return;
	}
	if (pcl::io::loadPCDFile(sceneFileName, *scene) < 0) {
		std::cout << "Error loading scene cloud" << std::endl;
		return;
	}
	recognition.recognize(model, scene);

	pcl::PointCloud<pcl::PointXYZRGB>::Ptr modelKeypoints = recognition.getModelKeypoints();
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr sceneKeypoints = recognition.getSceneKeypoints();
	std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > rototranslations = recognition.getRototranslations();
	std::vector<pcl::Correspondences> clustered_corrs = recognition.getClusteredCorrs();

	pcl::visualization::PCLVisualizer viewer("Correspondence Grouping");
	viewer.addPointCloud(scene, "scene_cloud");

	pcl::PointCloud<pcl::PointXYZRGB>::Ptr off_scene_model(new pcl::PointCloud<pcl::PointXYZRGB>());
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr off_scene_model_keypoints(new pcl::PointCloud<pcl::PointXYZRGB>());

	//  We are translating the model so that it doesn't end in the middle of the scene representation
	pcl::transformPointCloud(*model, *off_scene_model, Eigen::Vector3f(-1, 0, 0), Eigen::Quaternionf(1, 0, 0, 0));
	pcl::transformPointCloud(*modelKeypoints, *off_scene_model_keypoints, Eigen::Vector3f(-1, 0, 0), Eigen::Quaternionf(1, 0, 0, 0));

	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZRGB> off_scene_model_color_handler(off_scene_model, 255, 255, 128);
	viewer.addPointCloud(off_scene_model, off_scene_model_color_handler, "off_scene_model");

	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZRGB> scene_keypoints_color_handler(sceneKeypoints, 0, 0, 255);
	viewer.addPointCloud(sceneKeypoints, scene_keypoints_color_handler, "scene_keypoints");
	viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "scene_keypoints");

	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZRGB> off_scene_model_keypoints_color_handler(off_scene_model_keypoints, 0, 0, 255);
	viewer.addPointCloud(off_scene_model_keypoints, off_scene_model_keypoints_color_handler, "off_scene_model_keypoints");
	viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "off_scene_model_keypoints");

	for (size_t i = 0; i < rototranslations.size(); ++i)
	{
		pcl::PointCloud<pcl::PointXYZRGB>::Ptr rotated_model(new pcl::PointCloud<pcl::PointXYZRGB>());
		pcl::transformPointCloud(*model, *rotated_model, rototranslations[i]);

		std::stringstream ss_cloud;
		ss_cloud << "instance" << i;

		pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZRGB> rotated_model_color_handler(rotated_model, 255, 0, 0);
		viewer.addPointCloud(rotated_model, rotated_model_color_handler, ss_cloud.str());

		for (size_t j = 0; j < clustered_corrs[i].size(); ++j)
		{
			std::stringstream ss_line;
			ss_line << "correspondence_line" << i << "_" << j;
			pcl::PointXYZRGB& model_point = off_scene_model_keypoints->at(clustered_corrs[i][j].index_query);
			pcl::PointXYZRGB& scene_point = sceneKeypoints->at(clustered_corrs[i][j].index_match);

			//  We are drawing a line for each pair of clustered correspondences found between the model and the scene
			viewer.addLine<pcl::PointXYZRGB, pcl::PointXYZRGB>(model_point, scene_point, 0, 255, 0, ss_line.str());
		}
	}

	while (!viewer.wasStopped())
	{
		viewer.spinOnce();
	}
}

void captureModelAndSceneByKinect(char* modelFileName, char* sceneFileName) {
	const int screenWidth = 1280;
	const int screenHeight = 720;

	Kinect2Pcd kinect2Pcd;
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr scene;

	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("Point Cloud Viewer"));
	viewer->setSize(screenWidth, screenHeight);
	viewer->setCameraPosition(0.0, 0.0, -2.0, 0.0, 0.0, 0.0);
	viewer->registerMouseCallback(mouseEventOccurred, (void*)viewer.get());

	while (!viewer->wasStopped()) {
		viewer->spinOnce();

		if (kinect2Pcd.isUpdated()) {
			scene = kinect2Pcd.getPointCloud();
			if (!viewer->updatePointCloud(scene, "cloud")) {
				viewer->addPointCloud(scene, "cloud");
			}
		}
	}

	pcl::visualization::Camera camera;
	viewer->getCameraParameters(camera);
	camera.window_size[0] = screenWidth;
	camera.window_size[1] = screenHeight;

	pcl::PointCloud<pcl::PointXYZRGB>::Ptr model(new pcl::PointCloud<pcl::PointXYZRGB>());

	for (int i = 0; i < scene->size() ; i++) {
		pcl::PointXYZRGB pt = scene->at(i);
		if (pt.x != 0) {
			Eigen::Vector4d windowCord;
			camera.cvtWindowCoordinates(pt, windowCord);
			float screenX = windowCord[0];
			float screenY = windowCord[1];
			if (captureWindowMin.x() <= screenX && screenX <= captureWindowMax.x() && captureWindowMin.y() <= screenY && screenY <= captureWindowMax.y()) {
				model->push_back(scene->at(i));
			}
		}
	}

	pcl::io::savePCDFileASCII(sceneFileName, *scene);
	pcl::io::savePCDFileASCII(modelFileName, *model);
}

void mouseEventOccurred(const pcl::visualization::MouseEvent &event, void* viewerVoid) {
	pcl::visualization::PCLVisualizer* viewer = static_cast<pcl::visualization::PCLVisualizer *>(viewerVoid);

	if (event.getButton() == pcl::visualization::MouseEvent::RightButton && event.getType() == pcl::visualization::MouseEvent::MouseButtonPress) {
		std::cout << "clicked: " << event.getX() << " " << event.getY() << std::endl;
		if ((capturePointCnt++) % 2 == 0) {
			captureWindowMin = Eigen::Vector2f(event.getX(), event.getY());
		} else {
			captureWindowMax = Eigen::Vector2f(event.getX(), event.getY());
		}
	}
}

void merge2PointClouds(char* model1FileName, char* model2FileName)
{
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud1(new pcl::PointCloud<pcl::PointXYZRGB>());
	if (pcl::io::loadPCDFile(model1FileName, *cloud1) < 0) {
		std::cout << "Error loading model 1" << std::endl;
	}
	PointCloudProcess::mlsFiltering(cloud1);

	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud2(new pcl::PointCloud<pcl::PointXYZRGB>());
	if (pcl::io::loadPCDFile(model2FileName, *cloud2) < 0) {
		std::cout << "Error loading model 2" << std::endl;
	}
	PointCloudProcess::mlsFiltering(cloud2);

	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>());
	PointCloudProcess::merge2PointClouds(cloud, cloud1, cloud2);
	PointCloudProcess::mlsFiltering(cloud);
	/*pcl::PolygonMesh::Ptr mesh(new pcl::PolygonMesh());
	PointCloudProcess::pointCloud2Mesh(mesh, cloud);
	viewer.addPolygonMesh(*mesh, "model");*/

	pcl::visualization::PCLVisualizer viewer("Camera");
	viewer.addPointCloud(cloud, "model");
	viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "model");

	while (viewer.wasStopped() == false) {
		viewer.spinOnce();
	}
}
