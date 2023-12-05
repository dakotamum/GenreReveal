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
#include "readcsv.hpp"
#include "serialVerify.hpp"

using namespace std;

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

  std::vector<Point> implPoints(points_size); // Create a vector of the same size as the array
  std::copy(points_arr, points_arr + points_size, implPoints.begin());

  // Validate code with serial implementation
  kMeansClustering_serial(numEpochs, k, implPoints, points, centroids);

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
