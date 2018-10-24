#include "Timer.h"
#include "TeleCP.h"
#include <pcl/visualization/cloud_viewer.h>

TeleCP* telecp = NULL;

extern "C" {
	__declspec(dllexport) void callStart() {
		telecp = new TeleCP(true);
	}

	__declspec(dllexport) byte* callUpdate() {
		telecp->update();
		return telecp->getBuffer();
	}

	__declspec(dllexport) void callStop() {
		if (telecp != NULL) {
			delete telecp;
		}
	}
}
