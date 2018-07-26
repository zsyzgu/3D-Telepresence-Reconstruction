#ifndef TRANSMISSION_H
#define TRANSMISSION_H

#define _WINSOCK_DEPRECATED_NO_WARNINGS 
#include <Windows.h>
#include "TsdfVolume.cuh"


class Transmission {
private:
	const char* IP = "192.168.1.1";
	const int port = 1234;

	WSADATA wsaData;
	sockaddr_in sockAddr;
	SOCKET sock;

	bool isServer();
	void start(bool isServer);
	void sendData(char* data, int tot);
	void recvData(char* data, int tot);

	int delayFrames;
	char** buffer;
	char* sendBuffer;

public:
	Transmission(int delayFrames);
	~Transmission();
	void recvFrame();
	void sendFrame(int cameras, bool* check, float* depthImages_device, RGBQUAD* colorImages_device, Transformation* color2depth, Intrinsics* depthIntrinsics, Intrinsics* colorIntrinsics);
	int getFrame(float* depthImages_device, RGBQUAD* colorImages_device, Transformation* color2depth, Intrinsics* depthIntrinsics, Intrinsics* colorIntrinsics);
};

#endif
