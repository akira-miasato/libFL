#include <stdio.h>
#include <omp.h>
#include <vector>
#include <string>
#include <stdexcept>
#include <limits>
#include <utility>

#include "featureVector.h"
#include "bagOfVisualWords.h"

int moda(std::vector<int> vec){
    int max = 0;
    for(int i=0; i<vec.size; i++){
        if(vec[i] > max){
            max = vec[i];
        }
    }
    std::vector<int> n(max);
    for(int i=0; i<max; i++){
        n[i] = 0;
    }
    for(int i=0; i<vec.size; i++){
        n[vec[i]]++;
    }
    int ret = 0;
    int argret = 0;
    for(int i=0; i<max; i++){
        if(n[i] > argret){
            ret = i;
            arget = n[i];
        }
    }
    return ret;
}

bool comp_tuple(std::pair<double, int> i,
                std::pair<double, int> j){
    return(i.first < j.first);
}


std::vector<int> knn(FeatureMatrix* target, FeatureMatrix* trainX, std::vector<int> trainY){
    if(trainX->nFeaturesVectors != trainY.size()){
        throw std::runtime_error("X and Y from train ref mismatch!\n");
    }
    std::vector<int> ret;
    std::vector<float> nearest;
    int label;
    float d, di;
    for(int i=0; i<target->nFeaturesVectors; i++){
        d = std::numeric_limits<float>::max();
        for(int j=0; j<trainX->nFeaturesVectors; j++){
            di = vectorEuclideanDistance(
              trainX->featureVector[j],
              target->featureVector[i]
            );
            if (di < d){
                
            }
        }
    }
}


int main(int argc, char **argv) {
    using namespace std;
    double start_time = omp_get_wtime();

    int patchSize = 64;

    vector<string> data_dirs;
    // Each directory will have different labels assigned
    data_dirs.push_back("/home/akira-miasato/git/libFL/data/object6");
    data_dirs.push_back("/home/akira-miasato/git/libFL/data/object7");
    data_dirs.push_back("/home/akira-miasato/git/libFL/data/object8");

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
    

    start_time = omp_get_wtime();
    data_dirs.clear();
    data_dirs.push_back("/home/akira-miasato/git/libFL/data/object6dev");
    data_dirs.push_back("/home/akira-miasato/git/libFL/data/object7dev");
    data_dirs.push_back("/home/akira-miasato/git/libFL/data/object8dev");

    FeatureMatrix *featureMatrixDev = nullptr;
    std::vector<int> labelVectorDev;
    label = 0;
    for(int i=0; i<data_dirs.size(); i++) {
        path = data_dirs[i];
        directoryManager = loadDirectory(path.c_str(), 1);
        m1 = sampleHistograms(directoryManager, 128, 128, 1);
        for(int i=0; i<m1->nFeaturesVectors; i++) {
            labelVectorDev.push_back(label);
        }
        if(featureMatrixDev) {
            m2 = featureMatrixDev;
            featureMatrixDev = concatFeatureMatrices(m2, m1); // Preserve order
            destroyFeatureMatrix(&m1);
            destroyFeatureMatrix(&m2);
        }
        else {
            featureMatrixDev = m1;
        }
        destroyDirectoryManager(&directoryManager);
        label++;
    }

    time = omp_get_wtime() - start_time;
    printf("rows:%d cols:%d time:%f\n",
           featureMatrixDev->nFeaturesVectors,
           featureMatrixDev->featureVector[0]->size, time);

    // KMeansClustering
    start_time = omp_get_wtime();
    float loss;
    FeatureMatrix *dict = kMeansClustering(featureMatrix, 5, &loss);

    return 0;
}


