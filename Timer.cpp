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

float Timer::getTime() {
	LARGE_INTEGER en;
	QueryPerformanceCounter(&en);
	return (en.QuadPart - start) / (double)nFreq;
}

