#include <vector>
#include <iostream>
#include "csv.hpp"
#include "Point.hpp"

void kMeansClustering_serial(int epochs, int k, std::vector<Point> implPoints, std::vector<Point> origPoints, std::vector<Point> origCentroids) {
  for (int e = 0; e < epochs; e++)
  {
    for (std::vector<Point>::iterator c = begin(origCentroids); c != end(origCentroids); ++c) {
      // quick hack to get cluster index
      int clusterId = c - begin(origCentroids);

      for (std::vector<Point>::iterator it = origPoints.begin(); it != origPoints.end();
          ++it) {
        Point p = *it;
        double dist = c->distance(p);
        if (dist < p.minDist) {
          p.minDist = dist;
          p.cluster = clusterId;
        }
        *it = p;
      }
    }
    std::vector<int> nPoints;
    std::vector<double> sumX, sumY, sumZ;

    // Initialize with zeroes
    for (int j = 0; j < k; ++j) {
      nPoints.push_back(0);
      sumX.push_back(0.0);
      sumY.push_back(0.0);
      sumZ.push_back(0.0);
    }

    // Iterate over points to append data to centroids
    for (std::vector<Point>::iterator it = origPoints.begin(); it != origPoints.end(); ++it) {
      int clusterId = it->cluster;
      nPoints[clusterId] += 1;
      sumX[clusterId] += it->x;
      sumY[clusterId] += it->y;
      sumZ[clusterId] += it->z;

      it->minDist = __DBL_MAX__; // reset distance
    }

    // Compute the new centroids
    for (std::vector<Point>::iterator c = begin(origCentroids); c != end(origCentroids); ++c) {
      int clusterId = c - begin(origCentroids);
      c->x = sumX[clusterId] / nPoints[clusterId];
      c->y = sumY[clusterId] / nPoints[clusterId];
      c->z = sumZ[clusterId] / nPoints[clusterId];
    }
  }

  // Validate serial results with implementation
    int error_count = 0;
    for (int i = 0; i < implPoints.size(); i++)
    {
      if (implPoints[i].cluster != origPoints[i].cluster)
      {
        error_count++;
      }
    }
    printf("Verification with serial implementation ..........");
    if (error_count > 0) {
      printf(" FAIL\n");
      printf("%d out of %d point clusters do not match\n", error_count, origPoints.size());
    }
    else {
      printf(" PASS\n");
    }
}
