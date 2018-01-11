#include "Timer.h"
#include <Windows.h>

Timer::Timer()
{
	LARGE_INTEGER freq;
	QueryPerformanceFrequency(&freq);
	this->nFreq = freq.QuadPart;
}

Timer::~Timer()
{
}

void Timer::reset() {
	LARGE_INTEGER st;
	QueryPerformanceCounter(&st);
	start = st.QuadPart;
}

float Timer::getTime(int window) {
	LARGE_INTEGER en;
	QueryPerformanceCounter(&en);
	float thisFrame = (en.QuadPart - start) / (double)nFreq;
	record[pt] = thisFrame;
	pt = (pt + 1) % RECORD_N;
	if (window == 1) {
		return thisFrame;
	} else if (2 <= window && window <= RECORD_N) {
		float sum = 0;
		int t = pt - 1;
		for (int i = 0; i < window; i++) {
			if (--t < 0) {
				t += RECORD_N;
			}
			sum += record[t];
		}
		return sum / window;
	}
	return -1;
}
