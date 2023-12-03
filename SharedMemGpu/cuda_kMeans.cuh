#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Point.hpp"

#include <stdio.h>
#include <vector>

namespace Cuda_KMeans {
	void do_cuda_kMeans(int epochs, int k, char* category1,
                      char* category2, char* category3, 
                      Point* points, Point* centroids, int points_size, int centroids_size);
}
