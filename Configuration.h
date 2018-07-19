#ifndef CONFIGURATION_H
#define CONFIGURATION_H

#include "Parameters.h"
#include "TsdfVolume.cuh"

class Configuration {
public:
	static void saveExtrinsics(Transformation* transformation);
	static void loadExtrinsics(Transformation* transformation);
};

#endif
