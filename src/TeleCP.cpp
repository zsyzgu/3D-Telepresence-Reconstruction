#include "TeleCP.h"

TeleCP::TeleCP()
{
	cudaSetDevice(0);
	omp_set_num_threads(2);

	grabber = new RealsenseGrabber();
	volume = new TsdfVolume(2, 2, 2, 0, 0, 0);
	calibration = new Calibration();

	grabber->updateRGBD();

	if (transmission != NULL && transmission->isConnected) {
		transmission->prepareSendFrame(grabber, calibration->getExtrinsics());
	}
}

TeleCP::TeleCP(bool isServer)
{
	transmission = new Transmission(isServer);
	this->TeleCP::TeleCP();
}

TeleCP::~TeleCP()
{
	if (grabber != NULL) {
		delete grabber;
	}
	if (volume != NULL) {
		delete volume;
	}
	if (calibration != NULL) {
		delete[] calibration;
	}
	if (transmission != NULL) {
		delete transmission;
	}
	std::cout << "stopped" << std::endl;
}

void TeleCP::update()
{
#pragma omp parallel sections
	{
#pragma omp section
		{
			int remoteCameras = 0;
			if (transmission != NULL && transmission->isConnected) {
				transmission->sendFrame();
				remoteCameras = transmission->getFrame(grabber, calibration->getExtrinsics() + grabber->getCameras());
			}

			volume->integrate(grabber, remoteCameras, calibration->getExtrinsics());
			grabber->updateRGBD();

			if (transmission != NULL && transmission->isConnected) {
				transmission->prepareSendFrame(grabber, calibration->getExtrinsics());
			}
		}
#pragma omp section
		{
			if (transmission != NULL && transmission->isConnected) {
				transmission->recvFrame();
			}
		}
	}
}

void TeleCP::align()
{
	calibration->align(grabber);
}

void TeleCP::align(int targetId)
{
	calibration->align(grabber, targetId);
}

void TeleCP::setOrigin()
{
	calibration->setOrigin(grabber);
}

void TeleCP::saveBackground()
{
	grabber->saveBackground();
}
