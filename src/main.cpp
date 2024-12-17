#include "argparse.h"
#include "sequentialKmeans.h"
#include "kmeans_kernel.h"
#include "kmean_thrust.h"

int main(int argc, char **argv){
    struct options_t opts;
    get_opts(argc, argv, &opts);
    
    if(opts.parallel == 0){
        kmeansSeq(&opts);
    }
    if(opts.parallel == 1)
    {
        kmeans_cuda(&opts);
    }
    if(opts.parallel == 2){
        kmeans_cuda_shared(&opts);
    }
    if(opts.parallel == 3){
        kmeans_thrust(&opts);
    }
    return 0;
}