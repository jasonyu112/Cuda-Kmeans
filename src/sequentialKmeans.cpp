#include "sequentialKmeans.h"
#include "kmeans_kernel.h"

void * kmeansSeq(options_t* opts){
    //initialize and read inputs
    cudaEvent_t beginKmeans, stopKmeans;
    cudaEventCreate(&beginKmeans);
    cudaEventCreate(&stopKmeans);
    cudaEventRecord(beginKmeans);
    std::ifstream in;
    in.open(opts->in_file);
    int size = 0;   
    in >> size;
    in.close();
    int dims = opts->dims;
    int k = opts->num_clusters;
    double* input = (double*)malloc(size*dims*sizeof(double));
    double* center = (double*)malloc(k*dims*sizeof(double));

    read_file(opts, &size, input);
    
    //randomly generating centroids
    kmeans_srand(opts->seed);
    for(int i = 0;i<k;i++){
        int index = kmeans_rand()% size;
        std::memcpy(center+i*dims, input+index*dims, dims*sizeof(double));
    }

    
    int iterations = 0;
    int done = opts->max_iter;
    double oldCentroids[k][dims];
    std::vector<float*> timeVecs;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    while(iterations<done){
        cudaEventRecord(start);
        for(int i = 0;i<k;i++){
            std::memcpy(&oldCentroids[i], center+(i*dims), dims*sizeof(double));
        }
        
        //initialize labels as 2d arrays of size dims + 1
        //each datapoint in 2d array keeps the sum of all datapoints that map to it
        //extra 1 space will keep track of the number of times added
        //no_c is used to print when -c is not specified
        double labels[k][dims+1] = {};
        int no_c[size];

        for(int i = 0;i<size;i++){
            double* datapoint = input+i*dims;
            int new_index = -1;
            double smallest_dist = DBL_MAX;
            for(int k_index = 0;k_index<k;k_index++){
                double* centroid = center+k_index*dims;
                double e_dist = calculateEuclideanDistance(datapoint, centroid, dims);
                if (e_dist < smallest_dist){
                  smallest_dist = e_dist;
                  new_index = k_index;
                }
            }

            for (int index = 0; index<dims; index++){
                labels[new_index][index] += datapoint[index];
            }
            labels[new_index][dims] +=1;
            no_c[i] = new_index;
        }

        //new centroids are calculated as average of all centroids and placed in centers array
        for(int i = 0; i<k;i++){
            for(int j = 0;j<dims;j++){
                center[j+i*dims] = labels[i][j]/labels[i][dims];
            }
        }
        iterations+=1;

        //check for convergence
        if(converged((double*)&oldCentroids, center, opts)||iterations>=opts->max_iter){
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            float* time = (float*)malloc(sizeof(float));
            cudaEventElapsedTime(time, start, stop);
            timeVecs.push_back(time);

            double time_per_iter_in_ms = 0;
            for(unsigned int i =0;i<timeVecs.size();i++){
                time_per_iter_in_ms+=*timeVecs[0];
            }
            time_per_iter_in_ms= time_per_iter_in_ms/timeVecs.size();
            printf("%d,%lf\n", iterations, time_per_iter_in_ms);
            if (opts->control == false){
                printf("clusters:");
                for (int p=0; p < size; p++)
                    printf(" %d", no_c[p]);
            }
            else{
                for (int clusterId = 0; clusterId < k; clusterId ++){
                    printf("%d ", clusterId);
                    for (int d = 0; d < dims; d++)
                        printf("%lf ", center[clusterId*dims+d]);
                    printf("\n");
                }
            }
            break;
        }
        
        //timer
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float* time = (float*)malloc(sizeof(float));
        cudaEventElapsedTime(time, start, stop);
        timeVecs.push_back(time);
    }
    cudaEventRecord(stopKmeans);
    cudaEventSynchronize(stopKmeans);
    float temp;
    cudaEventElapsedTime(&temp, beginKmeans, stopKmeans);
    printf("%lf\n", temp);

    cudaEventDestroy(beginKmeans);
    cudaEventDestroy(stopKmeans);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    for(unsigned int i = 0; i<timeVecs.size();i++){
        free(timeVecs[i]);
    }
    free(center);
    free(input);
    return 0;
}