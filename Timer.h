#include <ctime>

class Timer
{
private:
	__int64 start;
	__int64 nFreq;
public:
	Timer();
	~Timer();
	void reset();
	float getTime();
};

