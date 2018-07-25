#include "Transmission.h"
#include <iostream>

char* Transmission::getHostIP()
{
	char hostName[256];
	gethostname(hostName, sizeof(hostName));
	HOSTENT* host = gethostbyname(hostName);
	return inet_ntoa(((in_addr*)*host->h_addr_list)[0]);
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

void Transmission::recvData(char* data, int tot)
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

Transmission::Transmission(bool isServer)
{
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
}

Transmission::Transmission()
{
	WSAStartup(MAKEWORD(2, 2), &wsaData);
	bool isServer = (strcmp(getHostIP(), IP) == 0);
	this->Transmission::Transmission(isServer);
}	

Transmission::~Transmission()
{
	closesocket(sock);
	WSACleanup();
}
