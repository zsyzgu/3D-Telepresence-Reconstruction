#include "Timer.h"

Timer::Timer()
{
}


Timer::~Timer()
{
}

void Timer::reset() {
	start = std::clock();
}

float Timer::getTime() {
	return (std::clock() - start) / (float)CLOCKS_PER_SEC;
}

