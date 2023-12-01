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

#include "cuda_kMeans.cuh"
#include "csv.hpp"

using namespace std;

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


    // Compute the new centroids
    for (std::vector<Point>::iterator c = begin(centroids); c != end(centroids); ++c) {
      int clusterId = c - begin(centroids);
      c->x = sumX[clusterId] / nPoints[clusterId];
      c->y = sumY[clusterId] / nPoints[clusterId];
      c->z = sumZ[clusterId] / nPoints[clusterId];
    }
  }
}


int main(int argv, char *argc[]) {
  int numEpochs = 5;
  int k = 3;
  char category1[] = "danceability";
  char category2[] = "loudness";
  char category3[] = "instrumentalness";

  printf("Reading CSV...\n");
  vector<Point> points = readcsv(category1, category2, category3);
  vector<Point> centroids;
  srand(100); // need to set the random seed
  for (int i = 0; i < k; ++i) {
    centroids.push_back(points.at(rand() % points.size()));
  }

  int points_size = points.size();
  int centroids_size = centroids.size();
  Point* points_arr = new Point[points_size];
  Point* centroids_arr = new Point[centroids.size()];

  copy(points.begin(),points.end(),points_arr);
  copy(centroids.begin(),centroids.end(),centroids_arr);

  // Call cuda wrapper
  Cuda_KMeans::do_cuda_kMeans(numEpochs, k, category1, category2, category3, 
            points_arr, centroids_arr, points_size, centroids_size);


  kMeansClustering_serial(numEpochs, k, points, centroids);

  // Validate serial results with cuda
  int error_count = 0;
  for (int i = 0; i < points_size; i++)
  {
    if (points[i].cluster != points_arr[i].cluster)
    {
      error_count++;
    }
  }
  if (error_count > 0)
    printf("%d out of %d point clusters do not match", error_count, points_size);
  else
    printf("Parrallel implementation verified with serial");

  printf("\n");

  // // Write output to file
  // ofstream myfile;
  // myfile.open("tracks_output.csv");
  // myfile << category1 << "," << category2 << "," << category3 << ",c" << endl;
  // for (int it = 0; it < points_size; it++) {
  //     myfile << points_arr[it].x << "," << points_arr[it].y << "," << points_arr[it].z << "," << points[it].cluster
  //         << endl;
  // }
  // myfile.close();

  delete[] points_arr;
  delete[] centroids_arr;
  return 0;
}
