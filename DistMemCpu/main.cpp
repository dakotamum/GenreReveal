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
#include <mpi.h>

#include "Config.hpp"
#include "readcsv.hpp"
#include "serialVerify.hpp"
#include "Point.hpp"
#include "timer.hpp"

int main(int argc, char *argv[]) {
  Config config;
  if (!config.parseInput(argc, argv))
    return 1;
  
  MPI_Init(NULL, NULL);
  int rank, size;

// create the MPI data type corresponding to the Point struct
  MPI_Datatype point_type;
  int lengths[5] = { 1, 1, 1, 1, 1 };
  MPI_Aint displacements[5];
  Point dummy_point;
  MPI_Aint base_address;
  MPI_Get_address(&dummy_point, &base_address);
  MPI_Get_address(&dummy_point.x, &displacements[0]);
  MPI_Get_address(&dummy_point.y, &displacements[1]);
  MPI_Get_address(&dummy_point.z, &displacements[2]);
  MPI_Get_address(&dummy_point.cluster, &displacements[3]);
  MPI_Get_address(&dummy_point.minDist, &displacements[4]);
  displacements[0] = MPI_Aint_diff(displacements[0], base_address);
  displacements[1] = MPI_Aint_diff(displacements[1], base_address);
  displacements[2] = MPI_Aint_diff(displacements[2], base_address);
  displacements[3] = MPI_Aint_diff(displacements[3], base_address);
  displacements[4] = MPI_Aint_diff(displacements[4], base_address);
  MPI_Datatype types[5] = { MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_INT, MPI_DOUBLE };
  MPI_Type_create_struct(5, lengths, displacements, types, &point_type);
  MPI_Type_commit(&point_type);

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // specified number of categories and epochs
  int k = config.k;
  int epochs = config.epochs;

  // declaration of vectors used in the various computations
  std::vector<int> sendCounts(size);
  std::vector<int> scatterDisplacements(size, 0);
  std::vector<int> globalClusterCounts(k, 0);
  std::vector<double> globalClusterSums(k*3, 0);
  std::vector<Point> origPoints;
  std::vector<Point> globalPoints;
  std::vector<Point> origCentroids;
  std::vector<Point> centroids(k);

  double startTime = get_wall_time();
  double endTime;
  double finalTime;
  double averageTime;
  if (rank == 0) {
    origPoints = readcsv(config.category1, config.category2, config.category3); // read from file
    // make copy of the points so we can use the original points for serial verification
    endTime = get_wall_time();
    finalTime = endTime -startTime; 
    printf("Reading data took %f\n", finalTime);

    globalPoints = origPoints;
    srand(100); // need to set the random seed

    // set centroids initially to random points
    for (int i = 0; i < k; ++i) {
      centroids[i] = globalPoints.at(rand() % globalPoints.size());
    }
    origCentroids = centroids;

    // calculate sendCounts and displacements to be used in the scatter operation
    int elementsPerProcess = globalPoints.size() / size;
    int remainingElements = globalPoints.size() % size;
    int displacement = 0;
    for (int i = 0; i < size; ++i) {
      sendCounts[i] = elementsPerProcess;
      if (i < remainingElements) {
          sendCounts[i]++;
      }
      scatterDisplacements[i] = displacement;
      displacement += sendCounts[i];
    }
  }
  // broadcast send counts, displacements, and centroids to all ranks
  MPI_Bcast(sendCounts.data(), size, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(scatterDisplacements.data(), size, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(centroids.data(), k, point_type, 0, MPI_COMM_WORLD);

  // get chunk of points to work on
  std::vector<Point> localPoints(sendCounts[rank]);
  MPI_Scatterv(globalPoints.data(), sendCounts.data(), scatterDisplacements.data(),
               point_type, localPoints.data(), sendCounts[rank], point_type, 0,
               MPI_COMM_WORLD);

  startTime = get_wall_time();
  for (int e = 0; e < epochs; e++)
  {
    // find cluster for each point
    for (std::vector<Point>::iterator c = begin(centroids); c != end(centroids); ++c) {
      // quick hack to get cluster index
      int clusterId = c - begin(centroids);
      // find minimum distance and assign point to a cluster accordingly
      for (std::vector<Point>::iterator it = localPoints.begin(); it != localPoints.end(); ++it) {
        Point p = *it;
        double dist = c->distance(p);
        if (dist < p.minDist) {
          p.minDist = dist;
          p.cluster = clusterId;
        }
        *it = p;
      }
    }

    // for storing local cluster counts and coordinate sums
    std::vector<int> localClusterCounts(k, 0);
    std::vector<double> localClusterSums(k*3, 0);

    // iterate over local points and add up cluster counts and sums
    for (std::vector<Point>::iterator it = localPoints.begin(); it != localPoints.end(); ++it) {
      Point p = *it;
      localClusterCounts[p.cluster]++;
      localClusterSums[p.cluster*3] += p.x;
      localClusterSums[p.cluster*3 + 1] += p.y;
      localClusterSums[p.cluster*3 + 2] += p.z;
      it->minDist = __DBL_MAX__; // reset distance
    }

    // reduce cluster counts and cluster distance sums to the global vectors
    MPI_Reduce(localClusterCounts.data(), globalClusterCounts.data(), k, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(localClusterSums.data(), globalClusterSums.data(), k * 3, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    // compute new cluster locations
    if (rank == 0) {
      for (int j = 0; j < k; ++j) {
        centroids[j].x = globalClusterSums[j * 3] / globalClusterCounts[j];
        centroids[j].y = globalClusterSums[j * 3 + 1] / globalClusterCounts[j];
        centroids[j].z = globalClusterSums[j * 3 + 2] / globalClusterCounts[j];
      }
    }

    // broadcast centroids with their updated locations
    MPI_Bcast(centroids.data(), k, point_type, 0, MPI_COMM_WORLD);
  }
  // unscatter the broken up global points vector
  MPI_Gatherv(localPoints.data(), sendCounts[rank], point_type, globalPoints.data(), sendCounts.data(), scatterDisplacements.data(), point_type, 0, MPI_COMM_WORLD);

  if (rank ==0){
    endTime = get_wall_time();
    double totalTime = endTime - startTime;
    averageTime = totalTime / epochs;
    printf("Algorithm took %f time to complete and averaged %f per epoch\n", totalTime, averageTime);

  }


  if (rank == 0) {
    // run serial verification
    kMeansClustering_serial(epochs, k, globalPoints, origPoints, origCentroids);
    
    // output resultant points with their assigned clusters to file
    if (config.writeToFile) {
      std::ofstream myfile;
      myfile.open("tracks_output.csv");
      myfile << config.category1 << "," << config.category2 << "," << config.category3 << ",c" << std::endl;

      for (std::vector<Point>::iterator it = globalPoints.begin(); it != globalPoints.end(); ++it) {
        myfile << it->x << "," << it->y << "," << it->z << "," << it->cluster
              << std::endl;
      }
      myfile.close();
    }
  }

  MPI_Finalize();
  return 0;
}
