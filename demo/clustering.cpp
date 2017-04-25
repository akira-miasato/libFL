#include <stdio.h>
#include <omp.h>
#include <vector>
#include <string>

#include "featureVector.h"
#include "bagOfVisualWords.h"

int main(int argc, char **argv) {
    using namespace std;
    double start_time = omp_get_wtime();

    int patchSize = 64;

    vector<string> data_dirs;
    // Each directory will have different labels assigned
    data_dirs.push_back("/home/valterf/git/libFL/data/object6");
    data_dirs.push_back("/home/valterf/git/libFL/data/object7");
    data_dirs.push_back("/home/valterf/git/libFL/data/object8");

    DirectoryManager* directoryManager;
    FeatureMatrix *featureMatrix = nullptr;
    FeatureMatrix *m1, *m2;
    string path;
    std::vector<int> labelVector;
    int label = 0;
    for(int i=0; i<data_dirs.size(); i++) {
        path = data_dirs[i];
        directoryManager = loadDirectory(path.c_str(), 1);
        m1 = sampleHistograms(directoryManager, 128, 128, 1);
        for(int i=0; i<m1->nFeaturesVectors; i++) {
            labelVector.push_back(label);
        }
        if(featureMatrix) {
            m2 = featureMatrix;
            featureMatrix = concatFeatureMatrices(m2, m1); // Preserve order
            destroyFeatureMatrix(&m1);
            destroyFeatureMatrix(&m2);
        }
        else {
            featureMatrix = m1;
        }
        destroyDirectoryManager(&directoryManager);
        label++;
    }

    double time = omp_get_wtime() - start_time;
    printf("rows:%d cols:%d time:%f\n",
           featureMatrix->nFeaturesVectors,
           featureMatrix->featureVector[0]->size, time);

    // KMeansClustering
    start_time = omp_get_wtime();
    float loss;
    FeatureMatrix *dict = kMeansClustering(featureMatrix, 5, &loss);

    float avg = 0;
    int k=0;
    start_time = omp_get_wtime();
    for(int i=0; i<dict->nFeaturesVectors; i++) {
        for(int j=i+1; j<dict->nFeaturesVectors; j++) {
            avg += vectorEuclideanDistance(
                        dict->featureVector[i],
                        dict->featureVector[j]
                    );
            k++;
        }
    }
    avg /= k;
    time = omp_get_wtime() - start_time;
    printf("Average centroid distance:%f; loss:%f; time:%f\n",
            avg, loss, time);
    destroyFeatureMatrix(&dict);

    return 0;
}


