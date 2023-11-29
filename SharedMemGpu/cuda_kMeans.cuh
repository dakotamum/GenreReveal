#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <vector>

struct Point {
  double x, y, z; // coordinates
  int cluster;    // no default cluster
  double minDist; // default infinite dist to nearest cluster

  Point() : x(0.0), y(0.0), z(0.0), cluster(-1), minDist(__DBL_MAX__) {}

  Point(double x, double y, double z)
      : x(x), y(y), z(z), cluster(-1), minDist(__DBL_MAX__) {}

  double distance(Point p) {
    return (p.x - x) * (p.x - x) + (p.y - y) * (p.y - y) +
           (p.z - z) * (p.z - z);
  }
};

namespace Cuda_KMeans {
	void do_cuda_kMeans(int epochs, int k, char* category1,
                      char* category2, char* category3, 
                      Point* points, Point* centroids, int points_size, int centroids_size);
}