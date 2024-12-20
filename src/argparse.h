#ifndef _ARGPARSE_H
#define _ARGPARSE_H

#include <getopt.h>
#include <stdlib.h>
#include <iostream>

struct options_t {
    char *in_file;
    char *out_file;
    int num_clusters;
    int dims;
    int max_iter;
    double threshhold;
    bool control;
    int seed;
    int parallel;
};

void get_opts(int argc, char **argv, struct options_t *opts);
#endif
