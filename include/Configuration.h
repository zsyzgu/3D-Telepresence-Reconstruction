#ifndef CONFIGURATION_H
#define CONFIGURATION_H

#include "Parameters.h"
#include "TsdfVolume.cuh"
#include "AlignColorMap.h"

class Configuration {
public:
	static void saveExtrinsics(Extrinsics* extrinsics);
	static void loadExtrinsics(Extrinsics* extrinsics);
	static void saveBackground(AlignColorMap* alignColorMap);
	static void loadBackground(AlignColorMap* alignColorMap);
	static int loadDelayFrame();
};

#endif
