#ifndef TRANSMISSION_H
#define TRANSMISSION_H

#define _WINSOCK_DEPRECATED_NO_WARNINGS 
#include <Windows.h>
#define BUFF_SIZE 4096
#define LEN 424 * 512

class Transmission {
private:
	//const char* IP = "192.168.1.1";
	const char* IP = "127.0.0.1";
	const int port = 1234;

	WSADATA wsaData;
	sockaddr_in sockAddr;
	SOCKET sock;

	UINT16* storedDepth;
	RGBQUAD* storedColor;

	char* getHostIP();
	void sendData(char* data, int tot);
	void recvData(char* data, int tot);

public:
	Transmission();
	~Transmission();
	void sendRGBD(UINT16* sendDepth, RGBQUAD* sendCoor);
	void recvRGBD(UINT16*& recvDepth, RGBQUAD*& recvColor);
};

#endif
