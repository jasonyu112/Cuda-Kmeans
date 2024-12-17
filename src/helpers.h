#include "argparse.h"
#include <iostream>
#include <fstream>
#include <math.h>
#include <map>
#include <vector>

void read_file(options_t* opts, int* size, double* input);
int kmeans_rand();
void kmeans_srand(unsigned int seed);
int converged(double* oldCentroids, double* centroids, const options_t* opts);
double calculateEuclideanDistance(double * c1, double * c2, int size);