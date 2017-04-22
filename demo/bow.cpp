#include "FL.h"
#include <omp.h>

int main(int argc, char **argv) {
    double start_time = omp_get_wtime();

    int patchSize = 8;
    DirectoryManager* directoryManager = loadDirectory("../data/objects", 1);
    FeatureMatrix* featureMatrix = computeHistogramPatches(directoryManager, patchSize);

    double time = omp_get_wtime() - start_time;
    printf("rows:%d cols:%d time:%f\n",featureMatrix->nFeaturesVectors,featureMatrix->featureVector[0]->size, time);
    destroyDirectoryManager(&directoryManager);
    return 0;
}


