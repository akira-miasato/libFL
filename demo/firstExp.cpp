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

FeatureVector* featFn(Image* img) {
    return computeHistogramForFeatureVector(img, 64, true);
//     return applyGranulometryOnImage(img, 5);
}

int main(int argc, char **argv) {
    using namespace std;
    /*
     * Some variable declarations which will be used through all experiments
     */
    FeatureMatrix *m1, *m2;
    string path;
    int label;
    int num_clusters = 10;
    double start_time, time;
    int knn_n = 10;
    int patchX, patchY;
    double kmeans_sampling = .01;
    double predict_sampling = .01;
    printf("patch\tacc\ttime\tsampling\n");
    for(int i=64; i > 8; i /= 2) {
        DirectoryManager* directoryManager;
        FeatureMatrix *featureMatrix = nullptr;
        vector<int> labelVector;
        patchX = patchY = i;

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
//         string root("/home/akira-miasato/git/libFL/data/");
//         train_dirs.push_back(root + "object6");
//         train_dirs.push_back(root + "object7");
//         train_dirs.push_back(root + "object8");

        string root("/home/akira-miasato/data/img/coil-100/");
        for(int i=1; i<=100; i++){
            train_dirs.push_back(root + "obj" + to_string(i));
        }

        FeatureMatrix *featureMatrixDev = nullptr;
        vector<int> labelVectorDev;
        vector<string> dev_dirs;
        for(string train_dir : train_dirs) {
            dev_dirs.push_back(train_dir + "dev");
        }

        /**
         * Extracting sampled patches training images.
         */
        start_time = omp_get_wtime();
        for(int i=0; i<train_dirs.size(); i++) {
            path = train_dirs[i];
            directoryManager = loadDirectory(path.c_str(), 1);
            m1 = computeHistogramPatches(directoryManager, patchX);
//             m1 = sampleFeatures(directoryManager, featFn, patchX, patchY, kmeans_sampling);
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

        /**
         * Extracting n centroids from training samples
         */
        float loss;
        FeatureMatrix *dict = kMeansClustering(featureMatrix, num_clusters, &loss);

        /**
         * Preparing feature matrix of VBOWs from training samples
         */
        destroyFeatureMatrix(&featureMatrix);
        label = 0;
        for(int i=0; i<train_dirs.size(); i++) {
            path = train_dirs[i];
            directoryManager = loadDirectory(path.c_str(), 1);
            m1 = patchHistSoftBow(directoryManager, dict, patchX);
//             m1 = sampleFeatureSoftBoW(directoryManager, dict, featFn, patchX, patchY, predict_sampling);
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

        label = 0;
        for(int i=0; i<dev_dirs.size(); i++) {
            path = dev_dirs[i];
            directoryManager = loadDirectory(path.c_str(), 1);
            m1 = patchHistSoftBow(directoryManager, dict, patchX);
//             m1 = sampleFeatureSoftBoW(directoryManager, dict, featFn, patchX, patchY, predict_sampling);
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

        std::vector<int> pred = knn(
                                    featureMatrixDev,
                                    featureMatrix,
                                    labelVector,
                                    knn_n,
                                    vectorCosineDistance
                                );

        double acc = 0;
        for(int i=0; i<pred.size(); i++) {
            if(pred[i] == labelVectorDev[i]) {
                acc++;
            }
        }
        acc /= pred.size();
        time = omp_get_wtime() - start_time;
        printf("%ix%i\t%f\t%f\t%f\n",
               patchX,
               patchY,
               acc,
               time,
               kmeans_sampling
              );
        kmeans_sampling *= 2;
        predict_sampling *= 2;
    }


    return 0;
}


