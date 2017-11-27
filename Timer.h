#include <ctime>

class Timer
{
private:
	std::clock_t start;
public:
	Timer();
	~Timer();
	void reset();
	float getTime();
};

