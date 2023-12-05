#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Point.hpp"

#include <stdio.h>
#include <vector>

namespace Cuda_KMeans {
	void do_cuda_kMeans(int k, Point* points, Point* centroids, int points_size, int centroids_size, 
                      int* nPoints,
                      double* sumX,
                      double* sumY,
                      double* sumZ);
}