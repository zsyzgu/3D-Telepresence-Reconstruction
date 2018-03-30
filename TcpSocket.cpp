#include "TcpSocket.h"


TcpSocket::~TcpSocket() {
	delete[] depthData;
	delete[] rgbData;

	delete[] revRgbData[0];
	delete[] revRgbData[1];

	delete[] revDepthData[0];
	delete[] revDepthData[1];
}

TcpSocket::TcpSocket() {
	curpos = 0;
	offset = 0;
	depthData = new UINT16[WIDTH * HEIGHT];
	rgbData = new RGBQUAD[WIDTH * HEIGHT];

	revDepthData[0] = new UINT16[WIDTH * HEIGHT];
	revRgbData[0] = new RGBQUAD[WIDTH * HEIGHT];
	revDepthData[1] = new UINT16[WIDTH * HEIGHT];
	revRgbData[1] = new RGBQUAD[WIDTH * HEIGHT];

}

void TcpSocket::SendDepthData() {
	std::cerr << "DEPTH: " << waitDepthLen << std::endl;
	sock = socket(PF_INET, SOCK_STREAM, 0);
	connect(sock, (SOCKADDR*)&sockAddr, sizeof(SOCKADDR));
	for (int i = 0; i < waitDepthLen; i += HBUF_SIZE) {
		int bs = HBUF_SIZE < (waitDepthLen - i) ? HBUF_SIZE : (waitDepthLen - i);
		for (int j = 0; j < bs; j++) {
			bufSend[j << 1] = (char)(depthData[i + j] >> 8);
			bufSend[j << 1 | 1] = (char)(depthData[i + j] & 255);
		}
		send(sock, bufSend, bs, 0);
	}
	waitDepthLen = 0;
	closesocket(sock);
}

void TcpSocket::SendRGBData() {
	std::cerr << "RGB: " << waitRgbLen << std::endl;
	sock = socket(PF_INET, SOCK_STREAM, 0);
	connect(sock, (SOCKADDR*)&sockAddr, sizeof(SOCKADDR));
	for (int i = 0; i < waitRgbLen; i += BUF_SIZE) {
		int bs = BUF_SIZE < (waitRgbLen - i) ? BUF_SIZE : (waitRgbLen - i);
		for (int j = 0; j < bs; j++)
			bufSend[j] = (char)rgbData[i + j].rgbRed;
		send(sock, bufSend, bs, 0);
		for (int j = 0; j < bs; j++)
			bufSend[j] = (char)rgbData[i + j].rgbGreen;
		send(sock, bufSend, bs, 0);
		for (int j = 0; j < bs; j++)
			bufSend[j] = (char)rgbData[i + j].rgbBlue;
		send(sock, bufSend, bs, 0);
	}
	waitRgbLen = 0;
	closesocket(sock);
}

int TcpSocket::getRemoteDepthAndColor(UINT16* depth, RGBQUAD* rgb) {
	depth = revDepthData[curpos ^ 1];
	rgb = revRgbData[curpos ^ 1];
	return 1;
}

void TcpSocket::Receive() {
	SOCKADDR clntAddr;
	int nsize = sizeof(SOCKADDR);
	int len;
	SOCKET csock = accept(sock, (SOCKADDR*)&clntAddr, &nsize);
	recv(csock, bufRecv, BUF_SIZE, 0);
	while (len = recv(csock, bufRecv, BUF_SIZE, 0)) {
		printf("%d\n", len);
		
		for (int i = 0; i < len; i++) {
			//RGB + DEPTH    WIDTH * HEIGHT * 3 + WIDTH * HEIGHT * 2
			offset++;
			if (offset == WIDTH * HEIGHT * 5) {
				offset = 0;
				curpos ^= 1;
			}
			if (offset < WIDTH * HEIGHT)
				revRgbData[curpos][offset].rgbRed = bufRecv[i];
			else if (offset >= WIDTH * HEIGHT && offset < WIDTH * HEIGHT * 2)
				revRgbData[curpos][offset - WIDTH * HEIGHT].rgbGreen = bufRecv[i];
			else if (offset >= WIDTH * HEIGHT * 2 && offset < WIDTH * HEIGHT * 3)
				revRgbData[curpos][offset - WIDTH * HEIGHT * 2].rgbBlue = bufRecv[i];
			else if (offset >= WIDTH * HEIGHT * 3 && offset < WIDTH * HEIGHT * 4)
				revDepthData[curpos][offset - WIDTH * HEIGHT * 3] = bufRecv[i];
			else if (offset >= WIDTH * HEIGHT * 4)
				revDepthData[curpos][offset - WIDTH * HEIGHT * 4] = revDepthData[curpos][offset] << 8 | bufRecv[i];
		}
	}
	closesocket(csock);
	printf("End\n");
}

void TcpSocket::SendData(UINT16* depth, int depthLen, RGBQUAD* rgb, int rgbLen) {
	printf("%d %d\n", depthLen, rgbLen);
	//#pragma omp critical
	{
		waitDepthLen = depthLen;
		for (int i = 0; i < depthLen; i++)
			depthData[i] = depth[i];
		waitRgbLen = rgbLen;
		for (int i = 0; i < rgbLen; i++)
			rgbData[i] = rgb[i];
	}
}

/*
void TcpSocket::SetDepthData(UINT16* data, int len) {
	//#pragma omp critical
	{
		waitDepthLen = len;
		for (int i = 0; i < len; i++)
			depthData[i] = data[i];
	}
}
void TcpSocket::SetRGBData(RGBQUAD* data, int len) {
	//#pragma omp critical
	{
		waitRgbLen = len;
		for (int i = 0; i < len; i++)
			rgbData[i] = data[i];
	}
}
*/