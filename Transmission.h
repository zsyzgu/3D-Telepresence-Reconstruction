#ifndef TRANSMISSION_H
#define TRANSMISSION_H

#define _WINSOCK_DEPRECATED_NO_WARNINGS 
#include <Windows.h>
#include "TsdfVolume.cuh"

class Transmission {
public:
	const char* IP = "192.168.1.1";
	const int port = 1288;

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
	int localFrames;
	int remoteFrames;

public:
	Transmission(int delayFrames);
	~Transmission();
	bool isConnected;
	void setDelayFrames(int delayFrames) { this->delayFrames = delayFrames; }
	void recvFrame();
	void sendFrame(int cameras, bool* check, float* depthImages_device, RGBQUAD* colorImages_device, Transformation* world2depth, Intrinsics* depthIntrinsics, Intrinsics* colorIntrinsics);
	int getFrame(float* depthImages_device, RGBQUAD* colorImages_device, Transformation* world2depth, Intrinsics* depthIntrinsics, Intrinsics* colorIntrinsics);
};

#endif
