#include "Configuration.h"
#include <fstream>
#include <iostream>

void Configuration::saveExtrinsics(Transformation* transformation)
{
	const char* EXTRINSICS_FILE = "Extrinsics.cfg";
	std::ofstream fout(EXTRINSICS_FILE);

	for (int i = 0; i < MAX_CAMERAS; i++) {
		fout << transformation->rotation0.x << " " << transformation->rotation0.y << " " << transformation->rotation0.z << std::endl;
		fout << transformation->rotation1.x << " " << transformation->rotation1.y << " " << transformation->rotation1.z << std::endl;
		fout << transformation->rotation2.x << " " << transformation->rotation2.y << " " << transformation->rotation2.z << std::endl;
		fout << transformation->translation.x << " " << transformation->translation.y << " " << transformation->translation.z << std::endl;
		transformation++;
	}

	std::cout << "Extrinsics saved." << std::endl;
}

void Configuration::loadExtrinsics(Transformation* transformation)
{
	const char* EXTRINSICS_FILE = "Extrinsics.cfg";
	std::fstream file;
	file.open(EXTRINSICS_FILE, std::ios::in);

	if (file) {
		std::ifstream fin(EXTRINSICS_FILE);

		for (int i = 0; i < MAX_CAMERAS; i++) {
			fin >> transformation->rotation0.x >> transformation->rotation0.y >> transformation->rotation0.z;
			fin >> transformation->rotation1.x >> transformation->rotation1.y >> transformation->rotation1.z;
			fin >> transformation->rotation2.x >> transformation->rotation2.y >> transformation->rotation2.z;
			fin >> transformation->translation.x >> transformation->translation.y >> transformation->translation.z;
			transformation++;
		}
	} else {
		for (int i = 0; i < MAX_CAMERAS; i++) {
			transformation->setIdentity();
			transformation++;
		}
	}
}
