#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "cuda_kMeans.cuh"
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "timer.hpp"

/**
 * Kernel:      initializeSums_kernel
 * 
 * In Args:     int centroids_size, int* nPoints, double* sumX, double* sumY, double* sumZ
 * Desc:        ONLY RUNS ON 1 THREAD, resets device values for nPoints, sumX, Y, and Z
*/
__global__ 
void initializeSums_kernel(int centroids_size, int* nPoints, double* sumX, double* sumY, double* sumZ)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx == 0)
    {
        // reset with zeroes
        for (int j = 0; j < centroids_size; j++) {
            nPoints[j] = 0;
            sumX[j] = 0;
            sumY[j] = 0;
            sumZ[j] = 0;
        }
    }
}

/**
 * Kernel:      kMeansClustering_kernel
 * 
 * In Args:     Point* points, Point* centroids, int points_size, int centroids_size, int* nPoints,
                double* sumX, double* sumY, double* sumZ
 * Desc:        Carries out the KMeansClustering on points, assigning a cluster to each point
*/
__global__
void kMeansClustering_kernel(Point* points, Point* centroids, int points_size, int centroids_size, int* nPoints,
                            double* sumX, double* sumY, double* sumZ)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    // Declare shared memory arrays for each block
    __shared__ int shared_nPoints[3];
    __shared__ double shared_sumX[3];
    __shared__ double shared_sumY[3];
    __shared__ double shared_sumZ[3];

    // Initialize shared memory for each cluster
    if (threadIdx.x < centroids_size) {
        shared_nPoints[threadIdx.x] = 0;
        shared_sumX[threadIdx.x] = 0.0;
        shared_sumY[threadIdx.x] = 0.0;
        shared_sumZ[threadIdx.x] = 0.0;
    }

    __syncthreads();

    if (idx < points_size) {
        for (int clusterId = 0; clusterId < centroids_size; clusterId++) {
            // quick hack to get cluster index
            Point c = centroids[clusterId];
            Point p = points[idx];
            // Distance() funtion brought in due to being __host__
            double dist = (p.x - c.x) * (p.x - c.x) + (p.y - c.y) * (p.y - c.y) +
                    (p.z - c.z) * (p.z - c.z);
            if (dist < p.minDist) {
                p.minDist = dist;
                p.cluster = clusterId;
            }
            points[idx] = p;
        }

        // Append data to shared memory in the block
        int clusterId = points[idx].cluster;
        atomicAdd(&shared_nPoints[clusterId], 1);
        atomicAdd(&shared_sumX[clusterId], points[idx].x);
        atomicAdd(&shared_sumY[clusterId], points[idx].y);
        atomicAdd(&shared_sumZ[clusterId], points[idx].z);

        points[idx].minDist = __DBL_MAX__; // reset distance
    }

    __syncthreads(); // Synchronize threads within the block before updating global memory

    // Update global memory with results from shared memory
    if (threadIdx.x < centroids_size) {
        atomicAdd(&nPoints[threadIdx.x], shared_nPoints[threadIdx.x]);
        atomicAdd(&sumX[threadIdx.x], shared_sumX[threadIdx.x]);
        atomicAdd(&sumY[threadIdx.x], shared_sumY[threadIdx.x]);
        atomicAdd(&sumZ[threadIdx.x], shared_sumZ[threadIdx.x]);
    }
}

/**
 * Kernel:      ResetClusters_kernel
 * 
 * In Args:     Point* centroids, int centroids_size, int* nPoints, double* sumX, double* sumY, double* sumZ
 * Desc:        ONLY RUNS 1 thread, selects new centroid points on device
*/
__global__
void ResetClusters_kernel(Point* centroids, int centroids_size, int* nPoints, double* sumX, double* sumY, double* sumZ)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx == 0)
    {
        // Compute the new centroids
        for (int clusterId = 0; clusterId < centroids_size; clusterId++) {
            centroids[clusterId].x = sumX[clusterId] / nPoints[clusterId];
            centroids[clusterId].y = sumY[clusterId] / nPoints[clusterId];
            centroids[clusterId].z = sumZ[clusterId] / nPoints[clusterId];
    }
    }
}

/**
 * wrapper:     Cuda_KMeans::do_cuda_kMeans
 * 
 * In Args:     int epochs, int k, char* category1,
                char* category2, char* category3, 
                Point* points, Point* centroids, int points_size, int centroids_size
 * Desc:        Setup for device, allocates cuda memory for GPU implementation of KMeansClustering
*/
namespace Cuda_KMeans {
	void do_cuda_kMeans(int epochs, int k, char* category1,
                      char* category2, char* category3, 
                      Point* points, Point* centroids, int points_size, int centroids_size)
	{

        // Allocate cuda memory for points and centroids
        Point* cuda_points, *cuda_centroids;
        cudaMalloc((void**)&cuda_points, points_size * sizeof(Point));
        cudaMalloc((void**)&cuda_centroids, centroids_size * sizeof(Point));
        cudaMemcpy(cuda_points, points, points_size * sizeof(Point), cudaMemcpyHostToDevice);

        int* cuda_nPoints;
        double* cuda_sumX, *cuda_sumY, *cuda_sumZ;
        cudaMalloc((void**)&cuda_nPoints, k * sizeof(int));
        cudaMalloc((void**)&cuda_sumX, k * sizeof(double));
        cudaMalloc((void**)&cuda_sumY, k * sizeof(double));
        cudaMalloc((void**)&cuda_sumZ, k * sizeof(double));

        int* nPoints = new int[k];
        double* sumX = new double[k];
        double* sumY = new double[k];
        double* sumZ = new double[k];

        // Initialize with zeroes
        for (int j = 0; j < k; j++) {
            nPoints[j] = 0;
            sumX[j] = 0;
            sumY[j] = 0;
            sumZ[j] = 0;
        }

        cudaMemcpy(cuda_centroids, centroids, centroids_size * sizeof(Point), cudaMemcpyHostToDevice);
        cudaMemcpy(cuda_nPoints, nPoints, k * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(cuda_sumX, sumX, k * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(cuda_sumY, sumY, k * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(cuda_sumZ, sumZ, k * sizeof(double), cudaMemcpyHostToDevice);

        int blockSize = 256;
        int numBlocks = (points_size + blockSize - 1) / blockSize;

	  double startTime = get_wall_time();
        for (int e = 0; e < epochs; e++)
        {
            printf("Cuda Epoch %d\n", e);
            // Launch Kernels
            initializeSums_kernel<<<1,1>>>(centroids_size, cuda_nPoints, cuda_sumX, cuda_sumY, cuda_sumZ);
            cudaDeviceSynchronize();
            kMeansClustering_kernel<<<numBlocks, blockSize>>>(cuda_points, cuda_centroids, points_size, centroids_size, cuda_nPoints, cuda_sumX, cuda_sumY, cuda_sumZ);
            cudaDeviceSynchronize();
            ResetClusters_kernel<<<1,1>>>(cuda_centroids, centroids_size, cuda_nPoints, cuda_sumX, cuda_sumY, cuda_sumZ);
            cudaDeviceSynchronize();
        }
	  double endTime = get_wall_time();
	  double totalTime = endTime - startTime;
	  double averageTime = totalTime / epochs;
	  printf("Algorithm took %f seconds to complete and averaged %f seconds per epoch\n", totalTime, averageTime);

        // Copy points memory from device to host
        cudaMemcpy(points, cuda_points, points_size * sizeof(Point), cudaMemcpyDeviceToHost);
        cudaMemcpy(centroids, cuda_centroids, points_size * sizeof(Point), cudaMemcpyDeviceToHost);
        cudaFree(cuda_points);
        cudaFree(cuda_centroids);
        cudaFree(cuda_nPoints);
        cudaFree(cuda_sumX);
        cudaFree(cuda_sumY);
        cudaFree(cuda_sumZ); 
	}
}
