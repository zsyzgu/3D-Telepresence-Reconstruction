#ifndef TRANSMISSION_H
#define TRANSMISSION_H

#define _WINSOCK_DEPRECATED_NO_WARNINGS 
#include <Windows.h>
#define BUFF_SIZE 4096
#define LEN 424 * 512

class Transmission {
private:
	const char* IP = "192.168.1.1";
	const int port = 1234;

	WSADATA wsaData;
	sockaddr_in sockAddr;
	SOCKET sock;

	UINT16* receivedDepth;
	RGBQUAD* receivedColor;

	char* getHostIP();

public:
	Transmission();
	~Transmission();
	void sendData(char* data, int tot);
	void receiveData(char* data, int tot);
	void sendRGBD(UINT16* sendDepth, RGBQUAD* sendCoor);
	void receiveRGBD(UINT16*& receivedDepth, RGBQUAD*& receivedColor);
};

#endif
