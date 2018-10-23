#include <ctime>

class Timer
{
	const static int RECORD_N = 100;
private:
	__int64 start;
	__int64 nFreq;
	int pt = 0;
	float record[RECORD_N];
public:
	Timer();
	~Timer();
	void reset();
	float getTime(int window = 1);
	void outputTime(int window = 1);
};
