#ifndef TRANSMISSION_H
#define TRANSMISSION_H

#define _WINSOCK_DEPRECATED_NO_WARNINGS 
#include <Windows.h>
#include "TsdfVolume.cuh"
#include "RealsenseGrabber.h"

#define MAX_DELAY_FRAME 20
#define FRAME_BUFFER_SIZE 30000000
#define BUFF_SIZE 16384

class RealsenseGrabber;

class Transmission {
public:
	const char* IP = "192.168.1.1";
	const int port = 1288;

	WSADATA wsaData;
	sockaddr_in sockAddr;
	SOCKET sock;

	//bool isServer();
	void start(bool isServer);
	void sendData(char* data, int tot);
	void recvData(char* data, int tot);

	int delayFrames;
	char** buffer;
	int sendOffset;
	char* sendBuffer;
	int localFrames;
	int remoteFrames;

public:
	Transmission(bool isServer, int delayFrames);
	~Transmission();
	bool isConnected;
	void setDelayFrames(int delayFrames) { this->delayFrames = delayFrames; }
	void recvFrame();
	void prepareSendFrame(bool* check, RealsenseGrabber* grabber, Transformation* extrinsics);
	void sendFrame();
	int getFrame(RealsenseGrabber* grabber, Transformation* extrinsics);
};

#endif
