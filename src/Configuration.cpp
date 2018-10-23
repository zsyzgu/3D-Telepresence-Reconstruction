#include "Configuration.h"
#include <fstream>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

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

	fout.close();
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

		fin.close();
	} else {
		for (int i = 0; i < MAX_CAMERAS; i++) {
			transformation->setIdentity();
			transformation++;
		}
	}

	file.close();
}

void Configuration::saveBackground(AlignColorMap* alignColorMap)
{
	const char* BACKGROUND_FILE = "Background.cfg";
	FILE* fout = fopen(BACKGROUND_FILE, "w");
	
	float* depth = new float[MAX_CAMERAS * DEPTH_W * DEPTH_H];
	int* color = new int[MAX_CAMERAS * COLOR_W * COLOR_H];

	if (alignColorMap->isBackgroundOn()) {
		fprintf(fout, "1\n");

		alignColorMap->copyBackground_device2host(depth, (RGBQUAD*)color);
		for (int i = 0; i < MAX_CAMERAS * DEPTH_W * DEPTH_H; i++) {
			fprintf(fout, "%f\n", depth[i]);
		}
		for (int i = 0; i < MAX_CAMERAS * COLOR_W * COLOR_H; i++) {
#if CALIBRATION == false
			fprintf(fout, "%d\n", color[i]);
#else
			int id = i % (COLOR_W * COLOR_H);
			int r = id / COLOR_W;
			int c = id % COLOR_W;
			if ((r & 1) == 0 && (c & 1) == 0) {
				fprintf(fout, "%d\n", color[i]);
			}
#endif
		}
	} else {
		fprintf(fout, "0\n");
	}

	delete[] depth;
	delete[] color;

	fclose(fout);

	std::cout << "Background saved." << std::endl;
}

void Configuration::loadBackground(AlignColorMap* alignColorMap)
{
	const char* BACKGROUND_FILE = "Background.cfg";
	std::fstream file;
	file.open(BACKGROUND_FILE, std::ios::in);

	if (file) {
		FILE* fin = fopen(BACKGROUND_FILE, "r");

		bool isRemoveBackground;
		fscanf(fin, "%d", &isRemoveBackground);

		if (isRemoveBackground) {
			float* depth = new float[MAX_CAMERAS * DEPTH_W * DEPTH_H];
			int* color = new int[MAX_CAMERAS * COLOR_W * COLOR_H];

			for (int i = 0; i < MAX_CAMERAS * DEPTH_W * DEPTH_H; i++) {
				fscanf(fin, "%f", &depth[i]);
			}
			for (int i = 0; i < MAX_CAMERAS * COLOR_W * COLOR_H; i++) {
				fscanf(fin, "%d", &color[i]);
			}
			alignColorMap->enableBackground();
			alignColorMap->copyBackground_host2device(depth, (RGBQUAD*)color);

			delete[] depth;
			delete[] color;
		} else {
			alignColorMap->disableBackground();
		}

		fclose(fin);
	} else {
		alignColorMap->disableBackground();
	}

	file.close();
}

int Configuration::loadDelayFrame()
{
	const char* DELAY_FILE = "Delay.cfg";
	std::fstream file;
	file.open(DELAY_FILE, std::ios::in);

	int result = 1;
	if (file) {
		FILE* fin = fopen(DELAY_FILE, "r");
		fscanf(fin, "%d", &result);
		if (result <= 0) {
			result = 1;
		}
		fclose(fin);
	}
	file.close();

	return result;
}
