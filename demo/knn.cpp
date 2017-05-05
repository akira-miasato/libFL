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

FeatureVector* histogramExtractor(Image* img){
    return computeHistogramForFeatureVector(img, 64, true);
}

int main(int argc, char **argv) {
    using namespace std;
    double start_time = omp_get_wtime();

    int patchSize = 128;

    string root("/home/akira-miasato/data/img/coil-100/");
    vector<string> train_dirs;
    for(int i=1; i<=100; i++){
        train_dirs.push_back(root + "obj" + to_string(i));
    }

    FeatureMatrix *featureMatrixDev = nullptr;
    vector<int> labelVectorDev;
    vector<string> dev_dirs;
    for(string train_dir : train_dirs){
        dev_dirs.push_back(train_dir + "dev");
    }

    DirectoryManager* directoryManager;
    FeatureMatrix *featureMatrix = nullptr;
    FeatureMatrix *m1, *m2;
    string path;
    std::vector<int> labelVector;
    int label = 0;
    for(int i=0; i<train_dirs.size(); i++) {
        path = train_dirs[i];
        directoryManager = loadDirectory(path.c_str(), 1);
        m1 = sampleFeatures(directoryManager, histogramExtractor, 128, 128, 1);
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

    label = 0;
    for(int i=0; i<dev_dirs.size(); i++) {
        path = dev_dirs[i];
        directoryManager = loadDirectory(path.c_str(), 1);
        m1 = sampleFeatures(directoryManager, histogramExtractor, 128, 128, 1);
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
    for(int i=1; i<16; i++){
        start_time = omp_get_wtime();
        std::vector<int> pred = knn(
            featureMatrixDev,
            featureMatrix,
            labelVector,
            i,
            vectorEuclideanDistance
        );

        double acc = 0;
        for(int i=0; i<pred.size(); i++){
            if(pred[i] == labelVectorDev[i]){
                acc++;
            }
        }
        acc /= pred.size();
        time = omp_get_wtime() - start_time;
        printf("%iNN acc:%f time:%f\n", i, acc, time);
    }

    return 0;
}


