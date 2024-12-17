#include "argparse.h"

void get_opts(int argc,
              char **argv,
              struct options_t *opts)
{
    if (argc == 1)
    {
        std::cout << "Usage:" << std::endl;
        std::cout << "\t--num_cluster or -k <int>" << std::endl;
        std::cout << "\t--dims or -d <int>" << std::endl;
        std::cout << "\t--inputfilename or -i <file_path>" << std::endl;
        std::cout << "\t--max_num_iter or -m <int>" << std::endl;
        std::cout << "\t--threshhold or -t <double>" << std::endl;
        std::cout << "\t--control or -c" << std::endl;
        std::cout << "\t--seed or -s <int>" << std::endl;
        std::cout << "\t--parallel or -p <int>" << std::endl;
        exit(0);
    }

    opts->max_iter = 150;
    opts->control = false;
    opts->parallel = 0;

    struct option l_opts[] = {
        {"inputfilename", required_argument, NULL, 'i'},
        {"num_cluser", required_argument, NULL, 'k'},
        {"dims", required_argument, NULL, 'd'},
        {"max_num_iter", no_argument, NULL, 'm'},
        {"threshhold", required_argument, NULL, 't'},
        {"control", no_argument, NULL, 'c'},
        {"seed", required_argument, NULL, 's'},
        {"parallel", required_argument, NULL, 'p'}
    };

    int ind, c;
    while ((c = getopt_long(argc, argv, "i:k:d:m:t:cs:p:", l_opts, &ind)) != -1)
    {
        switch (c)
        {
        case 0:
            break;
        case 'i':
            opts->in_file = (char *)optarg;
            break;
        case 'k':
            opts->num_clusters = atoi((char *)optarg);
            break;
        case 'd':
            opts->dims = atoi((char *)optarg);
            break;
        case 'm':
            opts->max_iter = atoi((char *)optarg);
            break;
        case 't':
            opts->threshhold = std::stof((char *)optarg);
            break;
        case 'c':
            opts->control = true;
            break;
        case 's':
            opts->seed = atoi((char *)optarg);
            break;
        case 'p':
            opts->parallel = atoi((char *)optarg);
            break;
        case ':':
            std::cerr << argv[0] << ": option -" << (char)optopt << "requires an argument." << std::endl;
            exit(1);
        }
    }
}
