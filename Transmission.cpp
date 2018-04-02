#include "Transmission.h"
#include <iostream>

char* Transmission::getHostIP()
{
	char hostName[256];
	gethostname(hostName, sizeof(hostName));
	HOSTENT* host = gethostbyname(hostName);
	char* hostIP;
	strcpy(hostIP, inet_ntoa(*(in_addr*)*host->h_addr_list));
	return hostIP;
}

Transmission::Transmission()
{
	WSAStartup(MAKEWORD(2, 2), &wsaData);
	bool isServer = (strcmp(getHostIP(), IP) == 0);
	std::cout << isServer << std::endl;

	memset(&sockAddr, 0, sizeof(sockAddr));
	sockAddr.sin_family = PF_INET;
	sockAddr.sin_addr.s_addr = inet_addr(IP);
	sockAddr.sin_port = htons(port);

	if (isServer) {
		//-- Server
		SOCKET sockSrv = socket(PF_INET, SOCK_STREAM, 0);
		bind(sockSrv, (SOCKADDR*)&sockAddr, sizeof(SOCKADDR));
		listen(sockSrv, 20);
		int len = sizeof(SOCKADDR);
		sock = accept(sockSrv, (SOCKADDR*)&sockAddr, &len);
	}
	else {
		//-- Client
		sock = socket(AF_INET, SOCK_STREAM, 0);
		connect(sock, (SOCKADDR*)&sockAddr, sizeof(sockAddr));
	}

	receivedDepth = new UINT16[LEN];
	receivedColor = new RGBQUAD[LEN];
}

Transmission::~Transmission()
{
	closesocket(sock);
	WSACleanup();

	if (receivedDepth != NULL) {
		delete[] receivedDepth;
	}
	if (receivedColor != NULL) {
		delete[] receivedColor;
	}
}

void Transmission::sendData(char* data, int tot)
{
	int offset = 0;
	while (offset != tot) {
		int len = min(BUFF_SIZE, tot - offset);
		send(sock, data + offset, len, 0);
		offset += len;
	}
}

void Transmission::receiveData(char* data, int tot)
{
	int len = 0;
	int offset = 0;
	while (len = recv(sock, data + offset, min(BUFF_SIZE, tot - offset), 0)) {
		offset += len;
		if (offset == tot) {
			break;
		}
	}
}

void Transmission::sendRGBD(UINT16* sendDepth, RGBQUAD* sendColor)
{
	sendData((char*)sendDepth, LEN * sizeof(UINT16));
	sendData((char*)sendColor, LEN * sizeof(RGBQUAD));
}


void Transmission::receiveRGBD(UINT16*& receivedDepth, RGBQUAD*& receivedColor)
{
	receivedDepth = this->receivedDepth;
	receivedColor = this->receivedColor;
	receiveData((char*)receivedDepth, LEN * sizeof(UINT16));
	receiveData((char*)receivedColor, LEN * sizeof(RGBQUAD));
}
