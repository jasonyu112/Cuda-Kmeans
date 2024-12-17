#include "kmeans_kernel.h"

int kmeans_cuda_shared(options_t * opts){
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

    //setup cuda device
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp,dev));
    CHECK(cudaSetDevice(dev));
    /*
    int blockSize;   // The launch block size
    int minGridSize; // The minimum grid size needed to achieve maximum occupancy
    int gridSize;    // The actual grid size to use, based on the device

    // Find the optimal launch parameters for maximum occupancy
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, mapDataShared, 0, 0);

    // Calculate launch grid size and shared memory
    gridSize = (deviceProp.multiProcessorCount * blockSize + deviceProp.warpSize - 1) / deviceProp.warpSize;

    printf("Optimal Block Size: %d\n", blockSize);
    printf("Grid Size: %d\n", gridSize);

    int activeWarpsPerSM;
    int maxWarpsPerSM = deviceProp.maxThreadsPerMultiProcessor / deviceProp.warpSize;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&activeWarpsPerSM, avgData, 32, 0);
    float occupancy = (float)activeWarpsPerSM / maxWarpsPerSM;
    printf("Occupancy: %.2f%%\n", occupancy * 100.0f);
    */
    /*
    printf("Number of multiprocessors: %d\n", deviceProp.multiProcessorCount);
    printf("Clock Rate (kHz): %d\n", deviceProp.clockRate);
    printf("Total amount of constant memory: %4.2f KB\n",deviceProp.totalConstMem/1024.0);
    printf("Total amount of shared memory per block: %4.2f KB\n",deviceProp.sharedMemPerBlock/1024.0);
    printf("Total number of registers available per block: %d\n",deviceProp.regsPerBlock);
    printf("Warp size: %d\n", deviceProp.warpSize);
    printf("Maximum number of threads per block: %d\n", deviceProp.maxThreadsPerBlock);
    printf("Maximum number of threads per multiprocessor: %d\n",deviceProp.maxThreadsPerMultiProcessor);
    printf("Maximum number of warps per multiprocessor: %d\n",deviceProp.maxThreadsPerMultiProcessor/32);
    cudaFuncAttributes attr;
    cudaFuncGetAttributes(&attr, mapDataShared);
    printf("Shared memory per block: %zu bytes\n", attr.sharedSizeBytes);
    */
    //int blockSize = 32;
    //int maxActiveBlocks =0;
    //cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxActiveBlocks, mapDataShared, blockSize, 0);
    //int blockCount = 68 * maxActiveBlocks;
    //printf("gridsize: %d\n", blockCount);
    
    //setup matrix size
    int input_x = size;
    int input_y = dims;
    int input_xy = input_x*input_y;
    int inputBytes = input_xy *sizeof(double);
    int centroid_x = k;
    int centroid_y = dims;
    int centroid_xy = centroid_x*centroid_y;
    int centroidBytes = centroid_xy*sizeof(double);
    int no_c_x = size;
    int no_c_Bytes = no_c_x*sizeof(int);
    int labels_Bytes = sizeof(double)*(k*(dims+1));
    double* device_inputs;
    double* device_centroids;
    int* device_no_c;
    double* device_labels;
    double* device_oldCentroid;
    int* device_converged;

    cudaMalloc((void**)&device_inputs, inputBytes);
    cudaMalloc((void**)&device_centroids, centroidBytes);
    cudaMalloc((void**)&device_no_c, no_c_Bytes);
    cudaMalloc((void**)&device_labels, labels_Bytes);
    cudaMalloc((void**)&device_oldCentroid, centroidBytes);
    cudaMalloc((void**)&device_converged, sizeof(int));

    //transfer inputs to device
    cudaEventRecord(memcpyWholeStart);
    cudaMemcpy(device_inputs, input, inputBytes, cudaMemcpyHostToDevice);
    cudaEventRecord(memcpyWholeStop);
    cudaEventSynchronize(memcpyWholeStop);
    cudaEventElapsedTime(&temp_total_memcpy_time, memcpyWholeStart, memcpyWholeStop);
    total_memcpy_time+=temp_total_memcpy_time;
    
    int iterations = 0;
    int done = opts->max_iter;
    int no_c[size];
    float memcpyTime = 0;
    std::vector<float*> timeVecs;

    cudaEventRecord(memcpyWholeStart);
    cudaMemcpy(device_centroids, center, centroidBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(device_oldCentroid, center, centroidBytes, cudaMemcpyHostToDevice);
    cudaEventRecord(memcpyWholeStop);
    cudaEventSynchronize(memcpyWholeStop);
    cudaEventElapsedTime(&temp_total_memcpy_time, memcpyWholeStart, memcpyWholeStop);
    total_memcpy_time+=temp_total_memcpy_time;

    cudaEvent_t start, stop, loop_memStart, loop_memStop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventCreate(&loop_memStart);
    cudaEventCreate(&loop_memStop);
    while(true){
        cudaEventRecord(start);
        double labels[k*(dims+1)] = {};
        cudaEventRecord(loop_memStart);
        cudaMemcpy(device_labels, labels, labels_Bytes, cudaMemcpyHostToDevice);
        cudaEventRecord(loop_memStop);
        cudaEventSynchronize(loop_memStop);
        float loop_memTime;
        cudaEventElapsedTime(&loop_memTime, loop_memStart, loop_memStop);
        memcpyTime+=loop_memTime;

        int block = 32;
        //int grid = 1088;
        int grid = (size+block-1)/block;
        int converged = 1;

        cudaEventRecord(loop_memStart);
        cudaMemcpy(device_converged, &converged, sizeof(int), cudaMemcpyHostToDevice);
        cudaEventRecord(loop_memStop);
        cudaEventSynchronize(loop_memStop);
        cudaEventElapsedTime(&loop_memTime, loop_memStart, loop_memStop);
        memcpyTime+=loop_memTime;

        int sharedMemSize = (k * (dims+1)) * sizeof(double);
        mapDataShared<<< grid, block,sharedMemSize >>>(device_inputs, device_centroids, device_labels, device_no_c, size, dims, k);

        block = 32;
        grid = ((k*dims)+block-1)/block;
        avgData<<< grid, block>>>(device_centroids, device_labels, dims, k);
        checkConvergence<<< grid, block>>>(device_oldCentroid, device_centroids, k, dims, opts->threshhold,device_converged);

        cudaEventRecord(loop_memStart);
        cudaMemcpy(&converged, device_converged, sizeof(int), cudaMemcpyDeviceToHost);
        cudaEventRecord(loop_memStop);
        cudaEventSynchronize(loop_memStop);
        cudaEventElapsedTime(&loop_memTime, loop_memStart, loop_memStop);
        memcpyTime+=loop_memTime;

        iterations+=1;
        if(converged || iterations>=done){
            cudaEventRecord(loop_memStart);
            cudaMemcpy(center, device_centroids, centroidBytes, cudaMemcpyDeviceToHost);
            if(opts->control == false){
                cudaMemcpy(no_c, device_no_c, no_c_Bytes, cudaMemcpyDeviceToHost);
            }
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
                time_per_iter_in_ms+=*timeVecs[i];
            }
            time_per_iter_in_ms= time_per_iter_in_ms/timeVecs.size();
            printf("%d,%lf\n", iterations, time_per_iter_in_ms);
            if(opts->control == false){
                printf("clusters:");
                for (int p=0; p < size; p++)
                    printf(" %d", no_c[p]);
            }else{
                for (int clusterId = 0; clusterId < k; clusterId ++){
                    printf("%d ", clusterId);
                    for (int d = 0; d < dims; d++)
                        printf("%lf ", center[clusterId*dims+d]);
                    printf("\n");
                }
            }
            break;
        }
        else{
            cudaEventRecord(loop_memStart);
            cudaMemcpy(device_oldCentroid, device_centroids, centroidBytes, cudaMemcpyDeviceToDevice);
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

    cudaFree(device_inputs);
    cudaFree(device_centroids);
    cudaFree(device_no_c);
    cudaFree(device_labels);
    cudaFree(device_oldCentroid);
    cudaFree(device_converged);
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
    free(input);
    free(center);
    total_memcpy_time += memcpyTime;
    printf("%lf,%lf,%lf\n", memcpyTime,total_memcpy_time,total_execution_time);
    return 0;
}

__global__ void mapDataShared(double* input, double* centroids, double* labels, int* no_c, int size, int dims, int k) {
    extern __shared__ double s_centroids[];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    // each thread loads a part of the centroids into shared memory
    #pragma unroll
    for (int i = tid; i < k * dims; i ++) {
        s_centroids[i] = centroids[i];
    }

    __syncthreads();

    if (idx < size) {
        double smallest_distance = DBL_MAX;
        int new_index = -1;

        for (int k_index = 0; k_index < k; k_index++) {
            double e_dist = 0;
            for (int j = 0; j < dims; j++) {
                double abs_diff = input[idx*dims+j]-s_centroids[k_index*dims+j];
                e_dist += abs_diff*abs_diff;
            }
            e_dist = e_dist;
            if (e_dist < smallest_distance) {
                smallest_distance = e_dist;
                new_index = k_index;
            }
        }

        for (int j = 0; j < dims; j++) {
            atomicAdd(&labels[new_index*(dims+1)+j], input[idx*dims+j]);
        }
        atomicAdd(&labels[new_index*(dims+1)+dims], 1);
        no_c[idx] = new_index;
    }
}

/*
__global__ void avgDataShared(double* centroids, double* labels, int dims, int k){
    extern __shared__ double sharedMem[];

    double* s_labels = sharedMem;

    int centroid_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    for(int i = tid; i<k*(dims+1);i++){
        s_labels[i]= labels[i];
    }
    __syncthreads();

    if (centroid_idx < k * dims){
        int k_index = centroid_idx / dims;
        int dim_index = centroid_idx %dims;
        int count = s_labels[k_index*(dims + 1) + dims];
        atomicExch((unsigned long long int*)&centroids[k_index * (dims)+dim_index], __double_as_longlong(s_labels[k_index*(dims+1)+dim_index]/count));
    }
}

__global__ void checkConvergenceShared(double* oldCentroids, double* centroids, int k, int dims, double threshhold, int* converged){
    //check for convergence between centroids and oldCentroids
    extern __shared__ double sharedMem[];

    double* s_oldCentroids = sharedMem;
    double* s_centroids = sharedMem+dims*k;

    int tid = threadIdx.x;

    for(int i = tid;i<k*dims;i++){
        s_oldCentroids[i] = oldCentroids[i];
        s_centroids[i] = centroids[i];
    }
    __syncthreads();

    int centroid_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(centroid_idx<k* dims){
        int k_index = centroid_idx / dims;
        int dim_index = centroid_idx%dims;
        double abs_diff = fabs(s_centroids[k_index*dims+dim_index]-s_oldCentroids[k_index*dims+dim_index]);
        if(abs_diff>(threshhold*threshhold)){
            *converged = 0;
        }
    }
}
*/