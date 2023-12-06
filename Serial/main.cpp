/*
This kmeans clustering algorithm was implemented using this tutorial:
https://reasonabledeviations.com/2019/10/02/k-means-in-cpp/. Modifications to
that code include the addition of a third dimension as well as reading and
parsing a different dataset.
*/

#include <ctime>
#include <fstream>
#include <iostream>
#include <vector>

#include "Config.hpp"
#include "Point.hpp"
#include "readcsv.hpp"
#include "timer.hpp"

void kMeansClustering(int epochs, int k, std::string category1,
                      std::string category2, std::string category3, bool writeToFile) {
  std::vector<Point> points = readcsv(category1, category2, category3); // read from file

  std::vector<Point> centroids;
  srand(100); // need to set the random seed
  for (int i = 0; i < k; ++i) {
    centroids.push_back(points.at(rand() % points.size()));
  }
  double startTime = get_wall_time();
  for (int e = 0; e < epochs; e++)
  {
    for (std::vector<Point>::iterator c = begin(centroids); c != end(centroids); ++c) {
      // quick hack to get cluster index
      int clusterId = c - begin(centroids);

      for (std::vector<Point>::iterator it = points.begin(); it != points.end();
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
    for (std::vector<Point>::iterator it = points.begin(); it != points.end(); ++it) {
      int clusterId = it->cluster;
      nPoints[clusterId] += 1;
      sumX[clusterId] += it->x;
      sumY[clusterId] += it->y;
      sumZ[clusterId] += it->z;

      it->minDist = __DBL_MAX__; // reset distance
    }

    // Compute the new centroids
    for (std::vector<Point>::iterator c = begin(centroids); c != end(centroids); ++c) {
      int clusterId = c - begin(centroids);
      c->x = sumX[clusterId] / nPoints[clusterId];
      c->y = sumY[clusterId] / nPoints[clusterId];
      c->z = sumZ[clusterId] / nPoints[clusterId];
    }
  }
  double endTime = get_wall_time();
  double totalTime = endTime - startTime;
  double averageTime = totalTime / epochs;
  printf("Algorithm took %f seconds to complete and averaged %f seconds per epoch\n", totalTime, averageTime);

  // write results to output file
  if (writeToFile) {
    std::ofstream myfile;
    myfile.open("tracks_output.csv");
    myfile << category1 << "," << category2 << "," << category3 << ",c" << std::endl;

    for (std::vector<Point>::iterator it = points.begin(); it != points.end(); ++it) {
      myfile << it->x << "," << it->y << "," << it->z << "," << it->cluster
            << std::endl;
    }
    myfile.close();
  }
}

int main(int argc, char *argv[]) {
  Config config; 
  if (!config.parseInput(argc, argv))
    return 1;

  int numEpochs = config.epochs;
  int k = config.k;
  kMeansClustering(numEpochs, k, config.category1, config.category2,
                   config.category3, config.writeToFile);
  return 0;
}
