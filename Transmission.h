#ifndef TRANSMISSION_H
#define TRANSMISSION_H

#define _WINSOCK_DEPRECATED_NO_WARNINGS 
#include <Windows.h>
#define BUFF_SIZE 1024

class Transmission {
private:
	const char* IP = "192.168.1.1";
	const int port = 1234;

	WSADATA wsaData;
	sockaddr_in sockAddr;
	SOCKET sock;

	char* getHostIP();
	void sendData(char* data, int tot);
	void recvData(char* data, int tot);

public:
	Transmission(bool isServer);
	Transmission();
	~Transmission();
};

#endif
