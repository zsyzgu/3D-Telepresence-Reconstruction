#include "Transmission.h"
#include "Parameters.h"
#include "Timer.h"
#include <iostream>

/*bool Transmission::isServer()
{
	char hostname[260];
	int ret = gethostname(hostname, sizeof(hostname));
	if (ret == -1) {
		return false;
	}
	struct hostent *host = gethostbyname(hostname);
	if (NULL == host) {
		return false;
	}
	in_addr* addr_list = (in_addr*)*host->h_addr_list;
	char* hostIP = inet_ntoa(addr_list[0]);
	return strcmp(hostIP, IP) == 0;
}*/

void Transmission::start(bool isServer)
{
	WSAStartup(MAKEWORD(2, 2), &wsaData);

	memset(&sockAddr, 0, sizeof(sockAddr));
	sockAddr.sin_family = PF_INET;
	sockAddr.sin_addr.s_addr = inet_addr(IP);
	sockAddr.sin_port = htons(port);

	if (isServer) {
		std::cout << "server" << std::endl;
		SOCKET sockSrv = socket(PF_INET, SOCK_STREAM, 0);
		bind(sockSrv, (SOCKADDR*)&sockAddr, sizeof(SOCKADDR));
		listen(sockSrv, 20);
		int len = sizeof(SOCKADDR);
		sock = accept(sockSrv, (SOCKADDR*)&sockAddr, &len);
	}
	else {
		std::cout << "client" << std::endl;
		sock = socket(AF_INET, SOCK_STREAM, 0);
		connect(sock, (SOCKADDR*)&sockAddr, sizeof(sockAddr));
	}

	isConnected = true;

	/*int nSendBuf = 128 * 1024 * 1024;
	setsockopt(sock, SOL_SOCKET, SO_SNDBUF, (const char*)&nSendBuf, sizeof(int));
	int nRecvBuf = 128 * 1024 * 1024;
	setsockopt(sock, SOL_SOCKET, SO_RCVBUF, (const char*)&nRecvBuf, sizeof(int));*/
}

void Transmission::sendData(char* data, int tot)
{
	int offset = 0;
	while (isConnected && offset != tot) {
		int len = min(BUFF_SIZE, tot - offset);
		int ret = send(sock, data + offset, len, 0);
		if (ret > 0) {
			offset += ret;
		}
		else if (ret == -1) {
			isConnected = false;
		}
	}
}

void Transmission::recvData(char* data, int tot)
{
	int offset = 0;
	while (isConnected && offset != tot) {
		int len = min(BUFF_SIZE, tot - offset);
		int ret = recv(sock, data + offset, len, 0);
		if (ret > 0) {
			offset += ret;
		}
		else if (ret == -1) {
			isConnected = false;
		}
	}
}

Transmission::Transmission(bool isServer, int delayFrames)
{
	start(isServer);
	
	this->delayFrames = delayFrames;
	localFrames = 0;
	remoteFrames = 0;

	buffer = new char*[MAX_DELAY_FRAME];
	for (int i = 0; i < MAX_DELAY_FRAME; i++) {
		buffer[i] = new char[FRAME_BUFFER_SIZE];
		memset(buffer[i], 0, FRAME_BUFFER_SIZE);
	}
	sendBuffer = new char[FRAME_BUFFER_SIZE];
	memset(sendBuffer, 0, FRAME_BUFFER_SIZE);
}	

Transmission::~Transmission()
{
	closesocket(sock);
	WSACleanup();

	if (buffer != NULL) {
		for (int i = 0; i < MAX_DELAY_FRAME; i++) {
			if (buffer[i] != NULL) {
				delete[] buffer[i];
			}
		}
		delete[] buffer;
	}
	if (sendBuffer != NULL) {
		delete[] sendBuffer;
	}
}

void Transmission::recvFrame()
{
	int len = 0;
	recvData((char*)(&len), sizeof(int));
	recvData(buffer[remoteFrames % MAX_DELAY_FRAME], len);
	remoteFrames++;
}

void Transmission::prepareSendFrame(int cameras, bool* check, float* depthImages_device, RGBQUAD* colorImages_device, Transformation* world2depth, Intrinsics* depthIntrinsics, Intrinsics* colorIntrinsics)
{
	sendOffset = 0;
	memcpy(sendBuffer + sendOffset, &cameras, sizeof(int));
	sendOffset += sizeof(int);
	memcpy(sendBuffer + sendOffset, check, cameras * sizeof(bool));
	sendOffset += cameras * sizeof(bool);
	for (int i = 0; i < cameras; i++) {
		if (check[i]) {
			cudaMemcpy(sendBuffer + sendOffset, depthImages_device + i * DEPTH_H * DEPTH_W, DEPTH_H * DEPTH_W * sizeof(float), cudaMemcpyDeviceToHost);
			sendOffset += DEPTH_H * DEPTH_W * sizeof(float);
			cudaMemcpy(sendBuffer + sendOffset, colorImages_device + i * COLOR_H * COLOR_W, COLOR_H * COLOR_W * sizeof(RGBQUAD), cudaMemcpyDeviceToHost);
			sendOffset += COLOR_H * COLOR_W * sizeof(RGBQUAD);
			memcpy(sendBuffer + sendOffset, world2depth + i, sizeof(Transformation));
			sendOffset += sizeof(Transformation);
			memcpy(sendBuffer + sendOffset, depthIntrinsics + i, sizeof(Intrinsics));
			sendOffset += sizeof(Intrinsics);
			memcpy(sendBuffer + sendOffset, colorIntrinsics + i, sizeof(Intrinsics));
			sendOffset += sizeof(Intrinsics);
		}
	}
}

void Transmission::sendFrame() {
	sendData((char*)(&sendOffset), sizeof(int));
	sendData(sendBuffer, sendOffset);
}

int Transmission::getFrame(float* depthImages_device, RGBQUAD* colorImages_device, Transformation* world2depth, Intrinsics* depthIntrinsics, Intrinsics* colorIntrinsics)
{
	if (localFrames - delayFrames < 0) {
		localFrames++;
		return 0;
	}
	int cameras = 0;
	bool check[MAX_CAMERAS];
	char* recvBuffer = buffer[(localFrames - delayFrames) % MAX_DELAY_FRAME];
	localFrames++;

	int offset = 0;
	memcpy(&cameras, recvBuffer + offset, sizeof(int));
	offset += sizeof(int);
	memcpy(check, recvBuffer + offset, cameras * sizeof(bool));
	offset += cameras * sizeof(bool);
	for (int i = 0; i < cameras; i++) {
		if (check[i]) {
			cudaMemcpy(depthImages_device + i * DEPTH_H * DEPTH_W, recvBuffer + offset, DEPTH_H * DEPTH_W * sizeof(float), cudaMemcpyHostToDevice);
			offset += DEPTH_H * DEPTH_W * sizeof(float);
			cudaMemcpy(colorImages_device + i * COLOR_H * COLOR_W, recvBuffer + offset, COLOR_H * COLOR_W * sizeof(RGBQUAD), cudaMemcpyHostToDevice);
			offset += COLOR_H * COLOR_W * sizeof(RGBQUAD);
			memcpy(world2depth + i, recvBuffer + offset, sizeof(Transformation));
			offset += sizeof(Transformation);
			memcpy(depthIntrinsics + i, recvBuffer + offset, sizeof(Intrinsics));
			offset += sizeof(Intrinsics);
			memcpy(colorIntrinsics + i, recvBuffer + offset, sizeof(Intrinsics));
			offset += sizeof(Intrinsics);
		}
	}

	return cameras;
}

