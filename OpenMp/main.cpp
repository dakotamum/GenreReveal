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
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include "csv.hpp"
#include <time.h>
#include <sys/time.h>

using namespace std;

double get_wall_time(){
    struct timeval time;
    if (gettimeofday(&time,NULL)){
        //  Handle error
        return 0;
    }
    return (double)time.tv_sec + (double)time.tv_usec * .000001;
}

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

vector<Point> readcsv(std::string cat1, std::string cat2, std::string cat3) {
  vector<Point> points;
  std::vector<Point> currentLine;
  csv::CSVReader reader("tracks_features.csv");
  for (csv::CSVRow &row : reader) {
    points.push_back(Point(row[cat1].get<double>(), row[cat2].get<double>(),
                           row[cat3].get<double>()));
  }
  return points;
}


void kMeansClustering_serial(int epochs, int k, vector<Point> &points, vector<Point> &centroids) {
  
  double startTime = get_wall_time();
  for (int e = 0; e < epochs; e++)
  {
    std::cout << "Serial epoch:" << e << std::endl;
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
  }
  double endTime = get_wall_time();
  double totalTime = endTime - startTime;
  double averageTime = totalTime / epochs;
  printf("Algorithm took %f time to complete and averaged %f per epoch\n", totalTime, averageTime);
}

void kMeansClustering(int epochs, int k, vector<Point> &points, vector<Point> &centroids, int thread_count) {

  double startTime = get_wall_time();
  for (int e = 0; e < epochs; e++)
  {
    std::cout << "OpenMp epoch:" << e << std::endl;
    for (int c = 0; c<centroids.size(); c++) {
      // quick hack to get cluster index
      int clusterId = c;
      Point centroid = centroids[c];

#pragma omp parallel for num_threads(thread_count)
      for (int i = 0; i<points.size();++i) {
        Point p = points[i];
        double dist = centroid.distance(p);
        if (dist < p.minDist) {
          p.minDist = dist;
          p.cluster = clusterId;
        }
        points[i] = p;
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
#pragma omp parallel for num_threads(thread_count) 
    for (int i = 0; i< points.size(); ++i) {
      int clusterId = points[i].cluster;
      nPoints[clusterId] += 1;
      sumX[clusterId] += points[i].x;
      sumY[clusterId] += points[i].y;
      sumZ[clusterId] += points[i].z;

      points[i].minDist = __DBL_MAX__; // reset distance
    }
  }
  double endTime = get_wall_time();
  double totalTime = endTime - startTime;
  double averageTime = totalTime / epochs;
  printf("Algorithm took %f time to complete and averaged %f per epoch\n", totalTime, averageTime);
}

int main(int argv, char *argc[]) {
  int thread_count = strtol(argc[1], NULL, 10);
  int numEpochs = 5;
  int k = 3;
  char category1[] = "danceability";
  char category2[] = "loudness";
  char category3[] = "instrumentalness";

  double wallStart = get_wall_time();
  vector<Point> points = readcsv(category1, category2, category3); // read from file
  vector<Point> centroids;
  double wallEnd = get_wall_time();

  printf("Program took %f s to load data\n", (wallEnd - wallStart));

  srand(100); // need to set the random seed
  for (int i = 0; i < k; ++i) {
    centroids.push_back(points.at(rand() % points.size()));
  }
  vector<Point> serial = points; 


  kMeansClustering(numEpochs, k, points, centroids, thread_count);
  kMeansClustering_serial(numEpochs, k, serial, centroids);

  // Validate serial results with cuda
  int error_count = 0;
  for (int i = 0; i < points.size(); i++)
  {
    if (points[i].cluster != serial[i].cluster)
    {
      error_count++;
    }
  }
  if (error_count > 0)
    printf("%d out of %d point clusters do not match", error_count, points.size());
  else
    printf("Parrallel implementation verified with serial");

  printf("\n");

  wallStart = get_wall_time();
  ofstream myfile;
  myfile.open("tracks_output.csv");
  myfile << category1 << "," << category2 << "," << category3 << ",c" << endl;

  for (vector<Point>::iterator it = points.begin(); it != points.end(); ++it) {
    myfile << it->x << "," << it->y << "," << it->z << "," << it->cluster
           << endl;
  }
  myfile.close();
  wallEnd = get_wall_time();
  printf("Program took %f s to write data\n", (wallEnd - wallStart));

  return 0;
}
