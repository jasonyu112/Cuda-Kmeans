#include <thrust/for_each.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/transform.h>
#include "kmeans_kernel.h"
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/for_each.h>


void kmeans_thrust(options_t * opts);
#ifdef __CUDACC__
struct MapData{
    double* centroids;
    double* count;
    double* labels;
    int* no_c;    
    int dims;
    int k;

    MapData(thrust::device_vector<double>& c,thrust::device_vector<double>& count, thrust::device_vector<double>& labels, thrust::device_vector<int>& no_c,
                                int dims, int k) 
        : centroids(thrust::raw_pointer_cast(c.data())), count(thrust::raw_pointer_cast(count.data())), labels(thrust::raw_pointer_cast(labels.data())),
            no_c(thrust::raw_pointer_cast(no_c.data())), dims(dims), k(k){}

    //input is passed in row by row
    //have edist calculated and compared
    __device__ void operator()(double* input, int index){
        double min_dist = DBL_MAX;
        int new_index = -1;

        for(int i = 0;i<k;i++){
            double e_dist = 0;

            for(int j =0;j<dims;j++){
                double point = input[index*dims+j];
                double centroidPoint = centroids[i*dims+j];
                e_dist += (point-centroidPoint)*(point-centroidPoint);
            }
            if (e_dist < min_dist){
                new_index = i;
                min_dist = e_dist;
            }
        }
        atomicAdd(&count[new_index],1);
        for(int i = 0;i<dims;i++){
            atomicAdd(&labels[new_index*dims+i], input[index*dims+i]);
        }
        no_c[index] = new_index;
    }
};

struct AvgData{
    double* d_label;
    double* d_centroids;

    AvgData(thrust::device_vector<double>& d_label, thrust::device_vector<double>& d_centroids)
        : d_label(thrust::raw_pointer_cast(d_label.data())), d_centroids(thrust::raw_pointer_cast(d_centroids.data())){}

    __device__ void operator()(int index, double* d_count, int k, int dims){
        int k_index = index / dims;
        int j = index % dims;
        double divisor = d_count[k_index];

        d_centroids[k_index*dims+j] = d_label[k_index*dims+j]/divisor;
    }
};

struct CheckConvergence{
    double* d_centroids;
    double* d_oldCentroids;
    int* flag;
    double threshhold;

    CheckConvergence(thrust::device_vector<double>& d_oldCentroids, thrust::device_vector<double>& d_centroids, int* flag, double threshhold)
        : d_centroids(thrust::raw_pointer_cast(d_centroids.data())), d_oldCentroids(thrust::raw_pointer_cast(d_oldCentroids.data())), flag(flag), threshhold(threshhold){}
    
    __device__ void operator()(int index, int dims){
        for(int i = 0;i<dims;i++){
            double abs_diff = fabs(d_centroids[index*dims+i]-d_oldCentroids[index*dims+i]);
            if (abs_diff>(threshhold*threshhold)){
                atomicExch(flag, 0);
            }
        }
    }
};
#endif