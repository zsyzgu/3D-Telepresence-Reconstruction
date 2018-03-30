#ifndef TCPSOCKET_H
#define TCPSOCKET_H

#define _WINSOCK_DEPRECATED_NO_WARNINGS
#include <Windows.h>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#define BUF_SIZE 4096
#define HBUF_SIZE 2048
#define WIDTH 512
#define HEIGHT 424

class TcpSocket {
public:
	TcpSocket();
	virtual ~TcpSocket();
	virtual void Init(const char* ip, int port) = 0;
	virtual void Run() = 0;
	void Clean() { WSACleanup(); }
	void SendDepthData();
	void SendRGBData();

	void Receive();
	int getRemoteDepthAndColor(UINT16* depth, RGBQUAD* rgb);
	//void SetDepthData(UINT16* data, int len);
	//void SetRGBData(RGBQUAD* data, int len);
	void SendData(UINT16* depth, int depthLen, RGBQUAD* rgb, int rgbLen);

protected:
	WSADATA wsaData;
	sockaddr_in sockAddr;
	SOCKET sock;

	char bufSend[BUF_SIZE];
	char bufRecv[BUF_SIZE];

	UINT16* depthData;
	RGBQUAD* rgbData;

	UINT16* revDepthData[2];
	RGBQUAD* revRgbData[2];

	int curpos;

	int waitDepthLen;
	int waitRgbLen;

	int offset;
};

class TcpClient : public TcpSocket {
public:
	TcpClient() {}
	void Init(const char* ip, int port) {
		WSAStartup(MAKEWORD(2, 2), &wsaData);
		memset(&sockAddr, 0, sizeof(sockAddr));
		sockAddr.sin_family = PF_INET;
		sockAddr.sin_addr.s_addr = inet_addr(ip);
		sockAddr.sin_port = htons(port);
		memset(bufSend, 0, sizeof(bufSend));
	}

	void Run() {
		if (waitRgbLen > 0)
			SendRGBData();
		if (waitDepthLen > 0)
			SendDepthData();
	}

};


class TcpServer : public TcpSocket {
public:
	TcpServer() {}
	void Init(const char* ip, int port) {
		WSAStartup(MAKEWORD(2, 2), &wsaData);
		memset(bufRecv, 0, sizeof(bufRecv));
		sock = socket(PF_INET, SOCK_STREAM, 0);
		memset(&sockAddr, 0, sizeof(sockAddr));
		sockAddr.sin_family = PF_INET;
		sockAddr.sin_addr.s_addr = inet_addr(ip);
		sockAddr.sin_port = htons(port);
		bind(sock, (SOCKADDR*)&sockAddr, sizeof(SOCKADDR));
		listen(sock, 20);
	}

	void Run() {
		puts("running!");
		Receive();
	}
};


#endif
