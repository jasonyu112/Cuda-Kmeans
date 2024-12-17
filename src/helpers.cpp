#include "helpers.h"
#include <iomanip>
static unsigned long int next = 1;
static unsigned long kmeans_rmax = 32767;

void read_file(options_t* opts, int* size,double* input){
    std::ifstream in;
    in.open(opts->in_file);

    in >> *size;

    int dimensions = opts->dims;
    
    for(int i = 0; i<*size;i++){
        int row = 0;
        in>> row;
        for(int j = 0; j<dimensions; j++){
            in>> input[i*dimensions+j];
        }
    }
    
    in.close();
}

int kmeans_rand() {
    next = next * 1103515245 + 12345;
    return (unsigned int)(next/65536) % (kmeans_rmax+1);
}

void kmeans_srand(unsigned int seed) {
    next = seed;
}


//checks if every datapoint is within the threshold
int converged(double* oldCentroids, double* centroids, const options_t* opts){
    int k = opts->num_clusters;
    int dims = opts->dims;
    double threshhold = opts->threshhold;

    for (int i = 0;i< k;i++){
        for(int j = 0;j< dims;j++){
            double abs_difference = fabs(oldCentroids[j+i*dims]-centroids[j+i*dims]);
            //double e_dist = calculateEuclideanDistance(oldCentroids[i*dims], centroids[i*dims], opts->dims);
            if(abs_difference>pow(threshhold,2)){
                return 0;
            }
        }
    }

    return 1;
}

double calculateEuclideanDistance(double * c1, double * c2, int size){
    double accum = 0;
    for(int i =0; i<size;i++){
        accum+= pow(c1[i]-c2[i],2);
    }
    return accum;
}