#include <iostream>
#include <stdio.h>
#include <omp.h>
#include <vector>
#include <algorithm>
#include <string>
#include <stdexcept>
#include <limits>
#include <utility>

#include "kNearestNeighbours.h"
#include "bagOfVisualWords.h"

int main(int argc, char **argv) {
    using namespace std;
    /*
     * Some variable declarations which will be used through all experiments
     */
    DirectoryManager* directoryManager;
    FeatureMatrix *featureMatrix = nullptr;
    vector<int> labelVector;
    FeatureMatrix *m1, *m2;
    string path;
    int label;
    int patchX=128, patchY=128;
    int num_clusters = 3;
    double kmeans_sampling = 1;
    double predict_sampling = 1;
    double start_time, time;
    
    vector<string> train_dirs;
    train_dirs.push_back("/home/akira-miasato/git/libFL/data/object6dev");
    train_dirs.push_back("/home/akira-miasato/git/libFL/data/object7dev");
    train_dirs.push_back("/home/akira-miasato/git/libFL/data/object8dev");
    
    FeatureMatrix *featureMatrixDev = nullptr;
    vector<int> labelVectorDev;
    vector<string> dev_dirs;
    dev_dirs.push_back("/home/akira-miasato/git/libFL/data/object6");
    dev_dirs.push_back("/home/akira-miasato/git/libFL/data/object7");
    dev_dirs.push_back("/home/akira-miasato/git/libFL/data/object8");
    
    /**
     * Extracting sampled patches training images.
     */
    start_time = omp_get_wtime();
    for(int i=0; i<train_dirs.size(); i++) {
        path = train_dirs[i];
        directoryManager = loadDirectory(path.c_str(), 1);
        m1 = sampleHistograms(directoryManager, patchX, patchY, kmeans_sampling);
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
    time = omp_get_wtime() - start_time;
    printf("rows:%d cols:%d time:%f\n",
           featureMatrix->nFeaturesVectors,
           featureMatrix->featureVector[0]->size, time);

    /**
     * Extracting n centroids from training samples
     */
    start_time = omp_get_wtime();
    float loss;
    FeatureMatrix *dict = kMeansClustering(featureMatrix, num_clusters, &loss);
    time = omp_get_wtime() - start_time;
    printf("rows:%d cols:%d loss:%f time:%f\n",
           dict->nFeaturesVectors,
           dict->featureVector[0]->size, loss, time);

    /**
     * Preparing feature matrix of VBOWs from training samples
     */
    start_time = omp_get_wtime();
    destroyFeatureMatrix(&featureMatrix);
    label = 0;
    for(int i=0; i<train_dirs.size(); i++){
        path = train_dirs[i];
        directoryManager = loadDirectory(path.c_str(), 1);
        m1 = sampleHistogramBoW(directoryManager, dict, patchX, patchY, predict_sampling);
        for(int i=0; i<m1->nFeaturesVectors; i++){
            labelVector.push_back(label);
        }
        if(featureMatrix){
            m2 = featureMatrix;
            featureMatrix = concatFeatureMatrices(m2, m1); // Preserve order
            destroyFeatureMatrix(&m1);
            destroyFeatureMatrix(&m2);
        }
        else{
            featureMatrix = m1;
        }
        destroyDirectoryManager(&directoryManager);
        label++;
    }
    time = omp_get_wtime() - start_time;
    printf("rows:%d cols:%d time:%f\n",
           featureMatrix->nFeaturesVectors,
           featureMatrix->featureVector[0]->size, time);

    /**
     * Preparing feature matrix of VBOWs from dev samples
     */
    start_time = omp_get_wtime();
    label = 0;
    for(int i=0; i<dev_dirs.size(); i++){
        path = dev_dirs[i];
        directoryManager = loadDirectory(path.c_str(), 1);
        m1 = sampleHistogramBoW(directoryManager, dict, patchX, patchY, predict_sampling);
        for(int i=0; i<m1->nFeaturesVectors; i++){
            labelVectorDev.push_back(label);
        }
        if(featureMatrixDev){
            m2 = featureMatrixDev;
            featureMatrixDev = concatFeatureMatrices(m2, m1); // Preserve order
            destroyFeatureMatrix(&m1);
            destroyFeatureMatrix(&m2);
        }
        else{
            featureMatrixDev = m1;
        }
        destroyDirectoryManager(&directoryManager);
        label++;
    }
    time = omp_get_wtime() - start_time;
    printf("rows:%d cols:%d time:%f\n",
           featureMatrixDev->nFeaturesVectors,
           featureMatrixDev->featureVector[0]->size, time);
    
    /**
     * Running KNN
     */
    start_time = omp_get_wtime();
    std::vector<int> pred = knn(
        featureMatrixDev,
        featureMatrix,
        labelVector,
        10,
        vectorCosineDistance
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

    
    return 0;
}


