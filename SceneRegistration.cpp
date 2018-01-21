#include "SceneRegistration.h"
#include "Timer.h"
//#include "pcl/keypoints/iss_3d.h"
//#include <pcl/filters/uniform_sampling.h>
#include <pcl/keypoints/sift_keypoint.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/registration/correspondence_rejection_distance.h>
#include <pcl/registration/correspondence_rejection_surface_normal.h>
#include <pcl/registration/correspondence_rejection_sample_consensus.h>

SceneRegistration::SceneRegistration() {

}

SceneRegistration::~SceneRegistration() {

}

void SceneRegistration::transform(const Eigen::Matrix4f & mat, const pcl::PointXYZ & p, pcl::PointXYZ & out) {
	out.x = mat(0, 0)*p.x + mat(0, 1)*p.y + mat(0, 2)*p.z + mat(0, 3);
	out.y = mat(1, 0)*p.x + mat(1, 1)*p.y + mat(1, 2)*p.z + mat(1, 3);
	out.z = mat(2, 0)*p.x + mat(2, 1)*p.y + mat(2, 2)*p.z + mat(2, 3);
}

Eigen::Matrix4f SceneRegistration::align(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr source, pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr target)
{
	Eigen::Matrix4f transformation;
	transformation.setIdentity();

	pcl::PointCloud<pcl::PointXYZ>::Ptr sourcePoints(new pcl::PointCloud<pcl::PointXYZ>());
	pcl::PointCloud<pcl::PointXYZ>::Ptr targetPoints(new pcl::PointCloud<pcl::PointXYZ>());
	pcl::copyPointCloud(*source, *sourcePoints);
	pcl::copyPointCloud(*target, *targetPoints);

	pcl::PointCloud<pcl::PointXYZRGB>::Ptr sourcePointsColor(new pcl::PointCloud<pcl::PointXYZRGB>());
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr targetPointsColor(new pcl::PointCloud<pcl::PointXYZRGB>());
	pcl::copyPointCloud(*source, *sourcePointsColor);
	pcl::copyPointCloud(*target, *targetPointsColor);

	pcl::PointCloud<pcl::Normal>::Ptr sourceNormals(new pcl::PointCloud<pcl::Normal>());
	pcl::PointCloud<pcl::Normal>::Ptr targetNormals(new pcl::PointCloud<pcl::Normal>());
	pcl::copyPointCloud(*source, *sourceNormals);
	pcl::copyPointCloud(*target, *targetNormals);

	pcl::PointCloud<pcl::PointWithScale>::Ptr sourceKeypointsScale(new pcl::PointCloud<pcl::PointWithScale>());
	pcl::PointCloud<pcl::PointWithScale>::Ptr targetKeypointsScale(new pcl::PointCloud<pcl::PointWithScale>());
	pcl::SIFTKeypoint<pcl::PointXYZRGB, pcl::PointWithScale> siftDetect;
	siftDetect.setScales(0.0025, 5, 5);
	siftDetect.setMinimumContrast(0.8);
	siftDetect.setInputCloud(sourcePointsColor);
	siftDetect.compute(*sourceKeypointsScale);
	siftDetect.setInputCloud(targetPointsColor);
	siftDetect.compute(*targetKeypointsScale);

	pcl::PointCloud<pcl::PointXYZRGB>::Ptr sourceKeypoints(new pcl::PointCloud<pcl::PointXYZRGB>());
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr targetKeypoints(new pcl::PointCloud<pcl::PointXYZRGB>());
	pcl::copyPointCloud(*sourceKeypointsScale, *sourceKeypoints);
	pcl::copyPointCloud(*targetKeypointsScale, *targetKeypoints);
	


	pcl::PointCloud<pcl::SHOT1344>::Ptr sourceDescr(new pcl::PointCloud<pcl::SHOT1344>());
	pcl::PointCloud<pcl::SHOT1344>::Ptr targetDescr(new pcl::PointCloud<pcl::SHOT1344>());
	pcl::SHOTColorEstimationOMP<pcl::PointXYZRGB, pcl::Normal, pcl::SHOT1344> descrEst;
	descrEst.setRadiusSearch(0.05);
	descrEst.setInputCloud(sourceKeypoints);
	descrEst.setSearchSurface(sourcePointsColor);
	descrEst.setInputNormals(sourceNormals);
	descrEst.compute(*sourceDescr);
	descrEst.setInputCloud(targetKeypoints);
	descrEst.setSearchSurface(targetPointsColor);
	descrEst.setInputNormals(targetNormals);
	descrEst.compute(*targetDescr);

	pcl::search::KdTree<pcl::SHOT1344> kdTree;
	kdTree.setInputCloud(targetDescr);

	pcl::CorrespondencesPtr corrs(new pcl::Correspondences());

	std::cout << sourceDescr->size() << std::endl;
//#pragma omp parallel for
	for (int i = 0; i < sourceDescr->size(); i++)
	{
		std::cout << i << std::endl;

		if (pcl_isnan(sourceDescr->at(i).descriptor[0])) {
			continue;
		}
		std::vector<int> targetIndex(2);
		std::vector<float> sqrDist(2);
		int found = kdTree.nearestKSearch(sourceDescr->points[i], 2, targetIndex, sqrDist);

		if (found == 2 && sqrDist[0] / sqrDist[1] < 0.64) {
			corrs->push_back(pcl::Correspondence(i, targetIndex[0], sqrDist[0]));
		}
	}

	pcl::PointCloud<pcl::PointXYZ>::Ptr sourceKeypointPoints(new pcl::PointCloud<pcl::PointXYZ>());
	pcl::PointCloud<pcl::PointXYZ>::Ptr targetKeypointPoints(new pcl::PointCloud<pcl::PointXYZ>());
	pcl::copyPointCloud(*sourceKeypoints, *sourceKeypointPoints);
	pcl::copyPointCloud(*targetKeypoints, *targetKeypointPoints);
	pcl::PointCloud<pcl::Normal>::Ptr sourceKeypointNormals(new pcl::PointCloud<pcl::Normal>());
	pcl::PointCloud<pcl::Normal>::Ptr targetKeypointNormals(new pcl::PointCloud<pcl::Normal>());
	pcl::copyPointCloud(*sourceKeypoints, *sourceKeypointNormals);
	pcl::copyPointCloud(*targetKeypoints, *targetKeypointNormals);


	pcl::registration::CorrespondenceRejectorSampleConsensus<pcl::PointXYZ> rejector;
	rejector.setInputSource(sourceKeypointPoints);
	rejector.setInputTarget(targetKeypointPoints);
	rejector.setInlierThreshold(0.01);
	rejector.setInputCorrespondences(corrs);
	rejector.getCorrespondences(*corrs);
	std::cout << corrs->size() << std::endl;

	estimateRigidTransformation(*sourceKeypointPoints, *targetKeypointPoints, *corrs, transformation);

	pcl::transformPointCloud(*source, *source, transformation);
	pcl::transformPointCloud(*sourceKeypoints, *sourceKeypoints, transformation);



	for (int i = 0; i < sourceKeypoints->size(); i++) {
		sourceKeypoints->points[i].r = sourceKeypoints->points[i].g = sourceKeypoints->points[i].b = 0;
		sourceKeypoints->points[i].r = 255;
	}
	for (int i = 0; i < targetKeypoints->size(); i++) {
		targetKeypoints->points[i].r = targetKeypoints->points[i].g = targetKeypoints->points[i].b = 0;
		targetKeypoints->points[i].g = 255;
	}

	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("Point Cloud Viewer"));
	viewer->setCameraPosition(0.0, 0.0, -2.0, 0.0, 0.0, 0.0);

	while (!viewer->wasStopped()) {
		viewer->spinOnce();

		pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>());

		pcl::copyPointCloud(*sourceKeypoints, *cloud);
		if (!viewer->updatePointCloud(cloud, "1")) {
			viewer->addPointCloud(cloud, "1");
		}
		pcl::copyPointCloud(*targetKeypoints, *cloud);
		if (!viewer->updatePointCloud(cloud, "2")) {
			viewer->addPointCloud(cloud, "2");
		}
		pcl::copyPointCloud(*source, *cloud);
		if (!viewer->updatePointCloud(cloud, "3")) {
			viewer->addPointCloud(cloud, "3");
		}
		pcl::copyPointCloud(*target, *cloud);
		if (!viewer->updatePointCloud(cloud, "4")) {
			viewer->addPointCloud(cloud, "4");
		}

		for (int i = 0; i < corrs->size(); i++) {
			viewer->addLine<pcl::PointXYZRGB, pcl::PointXYZRGB>(sourceKeypoints->points[(*corrs)[i].index_query], targetKeypoints->points[(*corrs)[i].index_match], 200, 200, 0, "line" + i);
		}
	}


	return transformation;
}

/*pcl::UniformSampling<pcl::PointXYZRGBNormal> uniformSampling;
pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr keypointsLocal(new pcl::PointCloud<pcl::PointXYZRGBNormal>());
pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr keypointsRemote(new pcl::PointCloud<pcl::PointXYZRGBNormal>());
uniformSampling.setRadiusSearch(0.02);
uniformSampling.setInputCloud(sceneLocal);
uniformSampling.filter(*keypointsLocal);
uniformSampling.setRadiusSearch(0.02);
uniformSampling.setInputCloud(sceneRemote);
uniformSampling.filter(*keypointsRemote);

pcl::PointCloud<pcl::Normal>::Ptr normalsLocal(new pcl::PointCloud<pcl::Normal>);
pcl::PointCloud<pcl::PointXYZ>::Ptr pointsLocal(new pcl::PointCloud<pcl::PointXYZ>);
pcl::copyPointCloud(*sceneLocal, *normalsLocal);
pcl::copyPointCloud(*sceneLocal, *pointsLocal);
pcl::PointCloud<pcl::Normal>::Ptr normalsRemote(new pcl::PointCloud<pcl::Normal>);
pcl::PointCloud<pcl::PointXYZ>::Ptr pointsRemote(new pcl::PointCloud<pcl::PointXYZ>);
pcl::copyPointCloud(*sceneRemote, *normalsRemote);
pcl::copyPointCloud(*sceneRemote, *pointsRemote);

pcl::PointCloud<pcl::SHOT352>::Ptr descrLocal(new pcl::PointCloud<pcl::SHOT352>);
pcl::PointCloud<pcl::SHOT352>::Ptr descrRemote(new pcl::PointCloud<pcl::SHOT352>);

pcl::SHOTEstimationOMP<pcl::PointXYZ, pcl::Normal, pcl::SHOT352> descrEst;
descrEst.setRadiusSearch(0.1);
descrEst.setInputCloud(pointsLocal);
descrEst.setInputNormals(normalsLocal);;
descrEst.compute(*descrLocal);

descrEst.setRadiusSearch(0.1);
descrEst.setInputCloud(pointsRemote);
descrEst.setInputNormals(normalsRemote);
descrEst.compute(*descrRemote);

pcl::SampleConsensusInitialAlignment<pcl::PointXYZRGBNormal, pcl::PointXYZRGBNormal, pcl::SHOT352> sac;
std::cout << keypointsLocal->size() << " " << descrLocal->size() << " " << keypointsRemote->size() << " " << descrRemote->size() << std::endl;
sac.setInputSource(keypointsLocal);
sac.setSourceFeatures(descrLocal);
sac.setInputTarget(keypointsRemote);
sac.setTargetFeatures(descrRemote);
sac.align(*result);
Eigen::Matrix4f transformation = sac.getFinalTransformation();*/
