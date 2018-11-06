#include "Configuration.h"
#include <fstream>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

void Configuration::saveExtrinsics(Extrinsics* extrinsics)
{
	const char* EXTRINSICS_FILE = "Extrinsics.cfg";
	std::ofstream fout(EXTRINSICS_FILE);

	for (int i = 0; i < MAX_CAMERAS; i++) {
		fout << extrinsics->rotation0.x << " " << extrinsics->rotation0.y << " " << extrinsics->rotation0.z << std::endl;
		fout << extrinsics->rotation1.x << " " << extrinsics->rotation1.y << " " << extrinsics->rotation1.z << std::endl;
		fout << extrinsics->rotation2.x << " " << extrinsics->rotation2.y << " " << extrinsics->rotation2.z << std::endl;
		fout << extrinsics->translation.x << " " << extrinsics->translation.y << " " << extrinsics->translation.z << std::endl;
		extrinsics++;
	}

	fout.close();
	std::cout << "Extrinsics saved." << std::endl;
}

void Configuration::loadExtrinsics(Extrinsics* extrinsics)
{
	const char* EXTRINSICS_FILE = "Extrinsics.cfg";
	std::fstream file;
	file.open(EXTRINSICS_FILE, std::ios::in);

	if (file) {
		std::ifstream fin(EXTRINSICS_FILE);

		for (int i = 0; i < MAX_CAMERAS; i++) {
			fin >> extrinsics->rotation0.x >> extrinsics->rotation0.y >> extrinsics->rotation0.z;
			fin >> extrinsics->rotation1.x >> extrinsics->rotation1.y >> extrinsics->rotation1.z;
			fin >> extrinsics->rotation2.x >> extrinsics->rotation2.y >> extrinsics->rotation2.z;
			fin >> extrinsics->translation.x >> extrinsics->translation.y >> extrinsics->translation.z;
			extrinsics++;
		}

		fin.close();
	} else {
		for (int i = 0; i < MAX_CAMERAS; i++) {
			extrinsics->setIdentity();
			extrinsics++;
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
#if HD == false
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
