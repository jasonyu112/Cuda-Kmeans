#include "kmean_thrust.h"

void kmeans_thrust(options_t * opts){
    cudaEvent_t wholeStart, wholeStop, memcpyWholeStart, memcpyWholeStop;
    cudaEventCreate(&wholeStart);
    cudaEventCreate(&wholeStop);
    cudaEventCreate(&memcpyWholeStart);
    cudaEventCreate(&memcpyWholeStop);
    float total_execution_time = 0;
    float total_memcpy_time = 0;
    float temp_total_memcpy_time;
    cudaEventRecord(wholeStart);

    //initialize and read inputs
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
        cudaEventRecord(memcpyWholeStart);
        std::memcpy(center+i*dims, input+index*dims, dims*sizeof(double));
        cudaEventRecord(memcpyWholeStop);
        cudaEventSynchronize(memcpyWholeStop);
        cudaEventElapsedTime(&temp_total_memcpy_time, memcpyWholeStart, memcpyWholeStop);
        total_memcpy_time+=temp_total_memcpy_time;
    }

    //copy input and center into thrust vectors
    thrust::device_vector<double> d_input(size*dims);
    thrust::device_vector<double> d_centroids(k*dims);
    thrust::device_vector<double> d_oldCentroids(k*dims);
    thrust::device_vector<double> d_labels(k*dims, 0);
    thrust::device_vector<double> d_count(k, 0);
    thrust::device_vector<int> d_no_c(size);

    //host vector for printing later
    thrust::host_vector<double> h_centroids;
    thrust::host_vector<double> h_no_c;

    cudaEventRecord(memcpyWholeStart);
    thrust::copy(input, input+size*dims, d_input.begin());
    thrust::copy(center, center+k*dims, d_centroids.begin());
    cudaEventRecord(memcpyWholeStop);
    cudaEventSynchronize(memcpyWholeStop);
    cudaEventElapsedTime(&temp_total_memcpy_time, memcpyWholeStart, memcpyWholeStop);
    total_memcpy_time+=temp_total_memcpy_time;

    int iterations = 0;
    int done = opts->max_iter;
    std::vector<float*> timeVecs;
    cudaEvent_t start, stop, loop_memStart, loop_memStop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventCreate(&loop_memStart);
    cudaEventCreate(&loop_memStop);
    float memcpyTime = 0;
    double* d_input_ptr = thrust::raw_pointer_cast(d_input.data());
    double* d_count_ptr = thrust::raw_pointer_cast(d_count.data());
    double* d_centroids_ptr = thrust::raw_pointer_cast(d_centroids.data());
    double* d_oldCentroids_ptr = thrust::raw_pointer_cast(d_oldCentroids.data());
    int* d_converged;
    cudaMalloc((void**)&d_converged, sizeof(int));
    while(true){
        cudaEventRecord(start);

        cudaEventRecord(loop_memStart);
        thrust::copy(d_centroids.begin(), d_centroids.end(), d_oldCentroids.begin()); //copying centroids into temp copy
        cudaEventRecord(loop_memStop);
        cudaEventSynchronize(loop_memStop);
        float loop_memTime;
        cudaEventElapsedTime(&loop_memTime, loop_memStart, loop_memStop);
        memcpyTime+=loop_memTime;

        //map datapoints to closest centroids
        MapData map_data(d_centroids, d_count, d_labels, d_no_c, dims, k);
        thrust::for_each(thrust::make_counting_iterator(0), thrust::make_counting_iterator(size),[=] __device__ (int index) mutable{
            map_data(d_input_ptr, index);
        });
        
        AvgData avg_data(d_labels, d_centroids);
        thrust::for_each(thrust::make_counting_iterator(0), thrust::make_counting_iterator(k*dims), [=] __device__ (int index) mutable{
            avg_data(index,d_count_ptr, k, dims);
        });

        CheckConvergence check_convergence(d_oldCentroids, d_centroids, d_converged, opts->threshhold);
        int h_converged = 1;
        cudaEventRecord(loop_memStart);
        cudaMemcpy(d_converged, &h_converged, sizeof(int), cudaMemcpyHostToDevice);
        cudaEventRecord(loop_memStop);
        cudaEventSynchronize(loop_memStop);
        cudaEventElapsedTime(&loop_memTime, loop_memStart, loop_memStop);
        memcpyTime+=loop_memTime;
        thrust::for_each(thrust::make_counting_iterator(0), thrust::make_counting_iterator(k),[=] __device__ (int index) mutable{
            check_convergence(index, dims);
        });

        cudaEventRecord(loop_memStart);
        cudaMemcpy(&h_converged, d_converged, sizeof(int), cudaMemcpyDeviceToHost);
        cudaEventRecord(loop_memStop);
        cudaEventSynchronize(loop_memStop);
        cudaEventElapsedTime(&loop_memTime, loop_memStart, loop_memStop);
        memcpyTime+=loop_memTime;

        iterations++;
        if(h_converged || iterations >= done){
            cudaEventRecord(loop_memStart);
            thrust::host_vector<double> h_centroids = d_centroids;
            thrust::host_vector<int> h_no_c = d_no_c;

            cudaEventRecord(loop_memStop);
            cudaEventSynchronize(loop_memStop);
            cudaEventElapsedTime(&loop_memTime, loop_memStart, loop_memStop);
            memcpyTime+=loop_memTime;

            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            float* time = (float*)malloc(sizeof(float));
            cudaEventElapsedTime(time, start, stop);
            timeVecs.push_back(time);

            double time_per_iter_in_ms = 0;
            for(unsigned int i =0;i<timeVecs.size();i++){
                time_per_iter_in_ms+= *timeVecs[i];
            }
            time_per_iter_in_ms= time_per_iter_in_ms/timeVecs.size();
            printf("%d,%lf\n", iterations, time_per_iter_in_ms);

            if(opts->control == false){
                printf("clusters:");
                for (int p=0; p < size; p++)
                    printf(" %d", h_no_c[p]);
            }
            else{
                for (int clusterId = 0; clusterId < k; clusterId ++){
                    printf("%d ", clusterId);
                    for (int d = 0; d < dims; d++)
                        printf("%lf ", h_centroids[clusterId*dims+d]);
                    printf("\n");
                }
            }
            break;
        }else{
            cudaEventRecord(loop_memStart);
            thrust::fill(d_labels.begin(), d_labels.end(), 0);
            thrust::fill(d_count.begin(), d_count.end(),0);
            cudaEventRecord(loop_memStop);
            cudaEventSynchronize(loop_memStop);
            cudaEventElapsedTime(&loop_memTime, loop_memStart, loop_memStop);
            memcpyTime+=loop_memTime;
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float* time = (float*)malloc(sizeof(float));
        cudaEventElapsedTime(time, start, stop);
        timeVecs.push_back(time);
    }
    cudaEventRecord(wholeStop);
    cudaEventSynchronize(wholeStop);
    cudaEventElapsedTime(&total_execution_time, wholeStart, wholeStop);

    cudaFree(d_converged);
    free(center);
    free(input);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaEventDestroy(loop_memStart);
    cudaEventDestroy(loop_memStop);
    cudaEventDestroy(wholeStart);
    cudaEventDestroy(wholeStop);
    cudaEventDestroy(memcpyWholeStart);
    cudaEventDestroy(memcpyWholeStop);

    for(int i = 0; i<timeVecs.size();i++){
        free(timeVecs[i]);
    }
    total_memcpy_time += memcpyTime;
    printf("%lf,%lf,%lf\n", memcpyTime,total_memcpy_time,total_execution_time);
}
