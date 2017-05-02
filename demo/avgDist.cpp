#include <stdio.h>
#include <omp.h>
#include <vector>
#include <string>

#include "histogram.h"
#include "featureVector.h"
#include "bagOfVisualWords.h"

FeatureVector* histogramExtractor(Image* img){
    return computeHistogramForFeatureVector(img, 64, true);
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
    for(int i=0; i<data_dirs.size(); i++){
        path = data_dirs[i];
        directoryManager = loadDirectory(path.c_str(), 1);
        m1 = sampleFeatures(directoryManager, histogramExtractor, 64, 64, 0.001);
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

    double time = omp_get_wtime() - start_time;
    printf("rows:%d cols:%d time:%f\n",
           featureMatrix->nFeaturesVectors,
           featureMatrix->featureVector[0]->size, time);
    
    int tot_in, tot_out;
    double avg_in, avg_out;
    
    tot_in = tot_out = avg_in = avg_out = 0;
    for(int i=0; i<featureMatrix->nFeaturesVectors; i++){
        for(int j=i+1; j<featureMatrix->nFeaturesVectors; j++){
            if(labelVector[i] == labelVector[j]){
                avg_in += vectorEuclideanDistance(
                    featureMatrix->featureVector[i],
                    featureMatrix->featureVector[j]                            
                );
                tot_in++;
            }
            else {
                avg_out += vectorEuclideanDistance(
                    featureMatrix->featureVector[i],
                    featureMatrix->featureVector[j]                            
                );
                tot_out++;
            }
            
        }
    }
    avg_in /= tot_in;
    avg_out /= tot_out;
    printf("average intra-class euclidean distance:%f\n", avg_in);  
    printf("average inter-class euclidean distance:%f\n", avg_out);


    tot_in = tot_out = avg_in = avg_out = 0;
    for(int i=0; i<featureMatrix->nFeaturesVectors; i++){
        for(int j=i+1; j<featureMatrix->nFeaturesVectors; j++){
            if(labelVector[i] == labelVector[j]){
                avg_in += vectorCosineDistance(
                    featureMatrix->featureVector[i],
                    featureMatrix->featureVector[j]                            
                );
                tot_in++;
            }
            else {
                avg_out += vectorCosineDistance(
                    featureMatrix->featureVector[i],
                    featureMatrix->featureVector[j]                            
                );
                tot_out++;
            }
            
        }
    }
    avg_in /= tot_in;
    avg_out /= tot_out;
    printf("average intra-class cosine distance:%f\n", avg_in);  
    printf("average inter-class cosine distance:%f\n", avg_out);
    

    tot_in = tot_out = avg_in = avg_out = 0;
    for(int i=0; i<featureMatrix->nFeaturesVectors; i++){
        for(int j=i+1; j<featureMatrix->nFeaturesVectors; j++){
            if(labelVector[i] == labelVector[j]){
                avg_in += vectorManhattanDistance(
                    featureMatrix->featureVector[i],
                    featureMatrix->featureVector[j]                            
                );
                tot_in++;
            }
            else {
                avg_out += vectorManhattanDistance(
                    featureMatrix->featureVector[i],
                    featureMatrix->featureVector[j]                            
                );
                tot_out++;
            }
            
        }
    }
    avg_in /= tot_in;
    avg_out /= tot_out;
    printf("average intra-class minkovsky distance:%f\n", avg_in);  
    printf("average inter-class minkovsky distance:%f\n", avg_out);
    
    return 0;
}


