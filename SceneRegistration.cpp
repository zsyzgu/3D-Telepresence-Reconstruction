#include "SceneRegistration.h"
#include "Timer.h"

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

	pcl::PointCloud<pcl::PointXYZ>::Ptr sourcePoints(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr targetPoints(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::copyPointCloud(*source, *sourcePoints);
	pcl::copyPointCloud(*target, *targetPoints);

	pcl::PointCloud<pcl::PointXYZRGB>::Ptr sourcePointsColor(new pcl::PointCloud<pcl::PointXYZRGB>);
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr targetPointsColor(new pcl::PointCloud<pcl::PointXYZRGB>);
	pcl::copyPointCloud(*source, *sourcePointsColor);
	pcl::copyPointCloud(*target, *targetPointsColor);

	pcl::PointCloud<pcl::Normal>::Ptr sourceNormals(new pcl::PointCloud<pcl::Normal>);
	pcl::PointCloud<pcl::Normal>::Ptr targetNormals(new pcl::PointCloud<pcl::Normal>);
	pcl::copyPointCloud(*source, *sourceNormals);
	pcl::copyPointCloud(*target, *targetNormals);

	pcl::PointCloud<pcl::SHOT1344>::Ptr sourceDescr(new pcl::PointCloud<pcl::SHOT1344>);
	pcl::PointCloud<pcl::SHOT1344>::Ptr targetDescr(new pcl::PointCloud<pcl::SHOT1344>);
	pcl::SHOTColorEstimationOMP<pcl::PointXYZRGB, pcl::Normal, pcl::SHOT1344> descrEst;
	descrEst.setRadiusSearch(0.01);
	descrEst.setInputCloud(sourcePointsColor);
	descrEst.setInputNormals(sourceNormals);
	descrEst.compute(*sourceDescr);
	descrEst.setInputCloud(targetPointsColor);
	descrEst.setInputNormals(targetNormals);
	descrEst.compute(*targetDescr);


	pcl::search::KdTree<pcl::SHOT1344> kdTree;
	kdTree.setInputCloud(targetDescr);

	pcl::Correspondences corrs;


#pragma omp parallel for
	for (int i = 0; i < sourcePoints->size(); i += 100)
	{
		if (pcl_isnan(sourceDescr->at(i).descriptor[0])) {
			continue;
		}
		std::vector<int> targetIndex(1);
		std::vector<float> sqrDist(1);
		kdTree.nearestKSearch(sourceDescr->points[i], 1, targetIndex, sqrDist);

		corrs.push_back(pcl::Correspondence(i, targetIndex[0], sqrDist[0]));
	}

	std::cout << corrs.size() << std::endl;

	std::sort(corrs.begin(), corrs.end(), SceneRegistration::compareCorrespondences);
	if (corrs.size() > 1000) {
		corrs.resize(1000);
	}

	for (int k = 0; k < corrs.size(); k++) {
		int i = corrs[k].index_query;
		int j = corrs[k].index_match;
		source->points[i].r = source->points[i].g = 255;
		source->points[i].b = 0;

		target->points[j].r = 0;
		target->points[j].g = target->points[j].b = 255;
	}

	estimateRigidTransformation(*sourcePoints, *targetPoints, corrs, transformation);

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
