#ifndef DEPTH_FILTER_H
#define DEPTH_FILTER_H

#include <iostream>
#include <thread>
#include <mutex>
#include <map>
#include <Windows.h>
#include <vector>

class DepthFilter {
public:
	DepthFilter();
	~DepthFilter();
	void process(UINT16* depthMap);
};

#endif
