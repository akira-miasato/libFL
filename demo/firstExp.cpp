#include <iostream>
#include <stdio.h>
#include <omp.h>
#include <vector>
#include <algorithm>
#include <string>
#include <stdexcept>
#include <limits>
#include <utility>

#include "morphology.h"
#include "kNearestNeighbours.h"
#include "bagOfVisualWords.h"

FeatureVector* featFn(Image* img){
    return computeHistogramForFeatureVector(img, 64, true);
//     return applyGranulometryOnImage(img, 10);
}

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
    int patchX=64, patchY=64;
    int num_clusters = 3;
    double kmeans_sampling = .1;
    double predict_sampling = .1;
    double start_time, time;
    int knn_max = 10;
    
    vector<string> train_dirs;
//     string root("/home/akira-miasato/data/img/cifar10_small/");
//     train_dirs.push_back(root + "airplane");
//     train_dirs.push_back(root + "automobile");
//     train_dirs.push_back(root + "bird");
//     train_dirs.push_back(root + "cat");
//     train_dirs.push_back(root + "deer");
//     train_dirs.push_back(root + "dog");
//     train_dirs.push_back(root + "frog");
//     train_dirs.push_back(root + "horse");
//     train_dirs.push_back(root + "ship");
//     train_dirs.push_back(root + "truck");
    string root("/home/akira-miasato/git/libFL/data/");
    train_dirs.push_back(root + "object6");
    train_dirs.push_back(root + "object7");
    train_dirs.push_back(root + "object8");

    FeatureMatrix *featureMatrixDev = nullptr;
    vector<int> labelVectorDev;
    vector<string> dev_dirs;
    for(string train_dir : train_dirs){
        dev_dirs.push_back(train_dir + "dev");
    }

    /**
     * Extracting sampled patches training images.
     */
    start_time = omp_get_wtime();
    for(int i=0; i<train_dirs.size(); i++) {
        path = train_dirs[i];
        directoryManager = loadDirectory(path.c_str(), 1);
        m1 = sampleFeatures(directoryManager, featFn, patchX, patchY, kmeans_sampling);
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
        m1 = sampleFeatureHardBoW(directoryManager, dict, featFn, patchX, patchY, .5, predict_sampling);
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
    for(float th=.05; th < 1.05; th += 0.05){
    start_time = omp_get_wtime();
    label = 0;
    for(int i=0; i<dev_dirs.size(); i++){
        path = dev_dirs[i];
        directoryManager = loadDirectory(path.c_str(), 1);
        m1 = sampleFeatureHardBoW(directoryManager, dict, featFn, patchX, patchY, th, predict_sampling);
//         m1 = sampleFeatureSoftBoW(directoryManager, dict, featFn, patchX, patchY, predict_sampling);
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
    std::cout << "Threshold: " << th << std::endl;
    for(int i=1; i<knn_max; i++){
        start_time = omp_get_wtime();
        std::vector<int> pred = knn(
            featureMatrixDev,
            featureMatrix,
            labelVector,
            i,
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
        printf("%iNN acc:\t%f\ttime:\t%f\n", i, acc, time);
    }
    }

    
    return 0;
}


