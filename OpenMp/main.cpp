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
#include "Point.hpp"
#include "readcsv.hpp"
#include "serialVerify.hpp"
#include <string.h>

using namespace std;

double get_wall_time(){
    struct timeval time;
    if (gettimeofday(&time,NULL)){
        //  Handle error
        return 0;
    }
    return (double)time.tv_sec + (double)time.tv_usec * .000001;
}


void kMeansClustering(int epochs, int k, vector<Point> &points, vector<Point> &centroids, int thread_count) {

    int i;
    int clusterId;
    int e;
    int j;
    int c;
    double dist;
    int thread; 
    int* nPoints = new int[k];
    double* sumX = new double[k];
    double* sumY = new double[k];
    double* sumZ = new double[k];
    Point centroid;
    Point p;
  double startTime = get_wall_time();
#pragma omp parallel num_threads(thread_count) default(none) shared(centroid, nPoints, sumX, sumY, sumZ, centroids, points, thread_count,k,epochs) private(p,thread, dist,c,j,e,clusterId,i)
  {
  for (e = 0; e < epochs; e++)
  {
    thread = omp_get_thread_num();
    if (thread == 0){
        printf("OpenMp epoch: %d\n", e);
    }
    for ( c = 0; c<centroids.size(); c++) {
      Point centroid = centroids[c];
#pragma omp for
      for (i = 0; i<points.size();++i) {
        Point p = points[i];
        dist = centroid.distance(p);
	if (c == 2 && i < 10){
		printf("ith %d dist: %f minDist: %f\n", i,dist, p.minDist);
	}
        if (dist < p.minDist) {
          p.minDist = dist;
          p.cluster = c;
        }
        points[i] = p;
      }
    }

#pragma omp for
    for (int i = 0; i < k; i++){
	nPoints[i] = 0;
	sumX[i] = 0.0;	
	sumY[i] = 0.0;	
	sumZ[i] = 0.0;	
    }
#pragma omp for reduction(+:sumX[:k]) reduction(+:sumY[:k]) reduction(+:sumZ[:k]) reduction(+:nPoints[:k])
    for (i = 0; i < points.size(); ++i) {
      clusterId = points[i].cluster;
      nPoints[clusterId] += 1;
      sumX[clusterId] = points[i].x;
      sumY[clusterId] = points[i].y;
      sumZ[clusterId] = points[i].z;
      points[i].minDist = __DBL_MAX__;
  }
  printf("0:%d 1:%d 2:%d \n",nPoints[0], nPoints[1], nPoints[2]);
#pragma omp for
    for (int i = 0; i<k; i++){
      clusterId = i;
      Point c = centroids[i];
      c.x = sumX[clusterId] / (float) nPoints[clusterId];
      c.y = sumY[clusterId] / (float) nPoints[clusterId];
      c.z = sumZ[clusterId] / (float) nPoints[clusterId];
      centroids[i] = c;
    }
  }
}
  double endTime = get_wall_time();
  double totalTime = endTime - startTime;
  double averageTime = totalTime / epochs;
  printf("Algorithm took %f time to complete and averaged %f per epoch\n", totalTime, averageTime);

  delete(nPoints);
  delete(sumX);
  delete(sumY);
  delete(sumZ);
}

int main(int argv, char *argc[]) {
  int thread_count = strtol(argc[1], NULL, 10);
  int numEpochs = 2;
  int k = 3;
  string category1 = "danceability";
  string category2 = "loudness";
  string category3 = "instrumentalness";

  printf("Reading csv...\n");
  double wallStart = get_wall_time();
  vector<Point> points = readcsv(category1, category2, category3); // read from file
  vector<Point> originalPoints = points;
  vector<Point> centroids;
  double wallEnd = get_wall_time();

  printf("Program took %f s to load data\n", (wallEnd - wallStart));

  srand(100); // need to set the random seed
  for (int i = 0; i < k; ++i) {
    centroids.push_back(points.at(rand() % points.size()));
  }
  vector<Point> originalCentroids = centroids;
  vector<Point> serial = points; 

  kMeansClustering(numEpochs, k, points, centroids, thread_count);
  kMeansClustering_serial(numEpochs, k, points, originalPoints, originalCentroids);

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
