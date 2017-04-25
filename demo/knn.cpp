#include <iostream>
#include <stdio.h>
#include <omp.h>
#include <vector>
#include <algorithm>
#include <string>
#include <stdexcept>
#include <limits>
#include <utility>

#include "featureVector.h"
#include "bagOfVisualWords.h"

int moda(std::vector<std::pair<float, int> > vec){
    int max = 0;
    for(int i=0; i<vec.size(); i++){
        if(vec[i].second > max){
            max = vec[i].second;
        }
    }
    max++;
    std::vector<int> n(max);
    for(int i=0; i<max; i++){
        n[i] = 0;
    }
    for(int i=0; i<vec.size(); i++){
        n[vec[i].second]++;
    }
    int ret = 0;
    int argret = 0;
    for(int i=0; i<max; i++){
        if(n[i] > argret){
            ret = i;
            argret = n[i];
        }
    }
    return ret;
}

bool comp_tuple(std::pair<float, int> i,
                std::pair<float, int> j){
    return(i.first < j.first);
}

std::vector<int> knn(FeatureMatrix* target, FeatureMatrix* trainX, std::vector<int> trainY, int k){
    if(trainX->nFeaturesVectors != trainY.size()){
        throw std::runtime_error("X and Y from train ref mismatch!\n");
    }
    std::vector<int> ret(target->nFeaturesVectors);
    std::vector<std::pair<float, int> > dist_pos(trainX->nFeaturesVectors);
    int label;
    float d, di;
    for(int i=0; i<target->nFeaturesVectors; i++){
        d = std::numeric_limits<float>::max();
        for(int j=0; j<trainX->nFeaturesVectors; j++){
            di = vectorEuclideanDistance(
              trainX->featureVector[j],
              target->featureVector[i]
            );
            dist_pos[j].first = di;
            dist_pos[j].second = trainY[j];
        }
        std::sort(dist_pos.begin(), dist_pos.end(), comp_tuple);
        std::vector<std::pair<float, int> > knearest(
            dist_pos.begin(), dist_pos.begin() + k  
        );
        ret[i] = moda(knearest);
    }
    return ret;
}

int main(int argc, char **argv) {
    using namespace std;
    double start_time = omp_get_wtime();

    int patchSize = 64;

    vector<string> data_dirs;
    // Each directory will have different labels assigned
    data_dirs.push_back("/home/valterf/git/libFL/data/object6dev");
    data_dirs.push_back("/home/valterf/git/libFL/data/object7dev");
    data_dirs.push_back("/home/valterf/git/libFL/data/object8dev");

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
    data_dirs.push_back("/home/valterf/git/libFL/data/object6");
    data_dirs.push_back("/home/valterf/git/libFL/data/object7");
    data_dirs.push_back("/home/valterf/git/libFL/data/object8");

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
    
    
    // KNN with raw histograms
     start_time = omp_get_wtime();
    std::vector<int> pred = knn(
        featureMatrixDev,
        featureMatrix,
        labelVector,
        1
    );

    double acc = 0;
    for(int i=0; i<pred.size(); i++){
        if(pred[i] == labelVectorDev[i]){
            acc++;
        }
    }
    acc /= pred.size();
    time = omp_get_wtime() - start_time;
    printf("KNN acc:%f time:%f\n", acc, time);

    // KMeansClustering
    start_time = omp_get_wtime();
    float loss;
    FeatureMatrix *dict = kMeansClustering(featureMatrix, 5, &loss);
    time = omp_get_wtime() - start_time;
    printf("rows:%d cols:%d time:%f\n",
           dict->nFeaturesVectors,
           dict->featureVector[0]->size, time);

    return 0;
}


