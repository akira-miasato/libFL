//
// Created by deangeli on 4/7/17.
//
#include <iostream>
#include <random>
#include <stdexcept>
#include <limits>
#include "kNearestNeighbours.h"
#include "featureVector.h"
#include "bagOfVisualWords.h"


FeatureMatrix* computeHistogramPatches(DirectoryManager* directoryManager, int patchSize){

    int binSize = 64;
    Image* firstImage = readImage(directoryManager->files[0]->path);
    int patchX_axis = firstImage->nx/patchSize;
    int patchY_axis = firstImage->ny/patchSize;
    int numberPatchsPerImage = patchX_axis*patchY_axis;
    int numberPatchs = numberPatchsPerImage*directoryManager->nfiles;
    //FeatureMatrix* featureMatrix = createFeatureMatrix();
    FeatureMatrix* featureMatrix = createFeatureMatrix(numberPatchs);
    destroyImage(&firstImage);
    int k=0;

// #pragma omp parallel for
    for (size_t fileIndex = 0; fileIndex < directoryManager->nfiles; ++fileIndex) {
        Image* currentImage = readImage(directoryManager->files[fileIndex]->path);

        for (int y = 0; y <= currentImage->ny-patchSize; y +=patchSize) {
            for (int x = 0; x <= currentImage->nx-patchSize; x += patchSize) {
                Image* patch = extractSubImage(currentImage,x,y,patchSize,patchSize,true);
                featureMatrix->featureVector[k] = computeHistogramForFeatureVector(patch,binSize,true);
                k++;
                destroyImage(&patch);
            }
        }
        destroyImage(&currentImage);
    }
    return featureMatrix;
}


FeatureMatrix* sampleFeatures(DirectoryManager* directoryManager,
                              FeatureExtractorFn featureExtractor,
                              int patch_x, int patch_y,
                              double sampling_factor,
                              int seed){
    srand(seed);
    Image* firstImage = readImage(directoryManager->files[0]->path);
    int patchX_axis = firstImage->nx - patch_x + 1;
    int patchY_axis = firstImage->ny - patch_y + 1;
    int numberPatchsPerImage = patchX_axis * patchY_axis;
    int numberPatchs = numberPatchsPerImage*directoryManager->nfiles;
    
    // Alocates enough memory, but final number of samples will be smaller
    FeatureMatrix* featureMatrix = createFeatureMatrix(numberPatchs * 2 * sampling_factor);

    destroyImage(&firstImage);
    int k = 0;
    double r;
    Image* patch;

// #pragma omp parallel for
    for (size_t fileIndex = 0; fileIndex < directoryManager->nfiles; ++fileIndex) {
        Image* currentImage = readImage(directoryManager->files[fileIndex]->path);
        for (int y = 0; y <= currentImage->ny - patch_y; ++y) {
            for (int x = 0; x <= currentImage->nx - patch_x; ++x) {
                r = (double)rand() / RAND_MAX; 
                if(r < sampling_factor){
                    patch = extractSubImage(currentImage, x, y,
                                            patch_x, patch_y, true);
                    featureMatrix->featureVector[k] = featureExtractor(patch);
                    k++;
                    destroyImage(&patch);
                }
            }
        }
        destroyImage(&currentImage);
    }

    // Smaller than allocated memory, but shouldn't be an issue
    featureMatrix->nFeaturesVectors = k;
    return featureMatrix;
}


FeatureMatrix* patchHistSoftBow(DirectoryManager* directoryManager,
                                FeatureMatrix* dictionary,
                                int patchSize){

    int binSize = 64;
    Image* firstImage = readImage(directoryManager->files[0]->path);
    int patchX_axis = firstImage->nx/patchSize;
    int patchY_axis = firstImage->ny/patchSize;
    int numberPatchsPerImage = patchX_axis*patchY_axis;
    int numberPatchs = numberPatchsPerImage*directoryManager->nfiles;
    //FeatureMatrix* featureMatrix = createFeatureMatrix();
    destroyImage(&firstImage);
    
    int k;
    int dim = dictionary->nFeaturesVectors;
    double r;
    Image* patch;
    FeatureVector* patch_hist = NULL;
    FeatureVector* patch_vbow = NULL;
    int word_index;
    
    FeatureMatrix* featureMatrix = createFeatureMatrix(directoryManager->nfiles);

// #pragma omp parallel for
    for (size_t fileIndex = 0; fileIndex < directoryManager->nfiles; ++fileIndex) {
        Image* currentImage = readImage(directoryManager->files[fileIndex]->path);
        featureMatrix->featureVector[fileIndex] = createFeatureVector(dim);
        for(int col=0; col<dim; col++){
            featureMatrix->featureVector[fileIndex]->features[col] = 0;
        }
        k = 0;
        for (int y = 0; y <= currentImage->ny-patchSize; y +=patchSize) {
            for (int x = 0; x <= currentImage->nx-patchSize; x += patchSize) {
                patch = extractSubImage(currentImage,x,y,patchSize,patchSize,true);
                if(patch_hist) destroyFeatureVector(&patch_hist);
                if(patch_vbow) destroyFeatureVector(&patch_vbow);
                patch_hist = computeHistogramForFeatureVector(patch,binSize,true);
                patch_vbow = computeSoftVBoW(patch_hist, dictionary);
                for(int col=0; col<dim; col++){
                    featureMatrix->featureVector[fileIndex]->features[col] +=
                        patch_vbow->features[col];
                }
                k++;
                destroyImage(&patch);
            }
        }
        for(int col=0; col<dim; col++){
            featureMatrix->featureVector[fileIndex]->features[col] /= k;
        }
        destroyImage(&currentImage);
    }
    return featureMatrix;
}




FeatureMatrix* patchHistBow(DirectoryManager* directoryManager,
                            FeatureMatrix* dictionary,
                            int patchSize){

    int binSize = 64;
    Image* firstImage = readImage(directoryManager->files[0]->path);
    int patchX_axis = firstImage->nx/patchSize;
    int patchY_axis = firstImage->ny/patchSize;
    int numberPatchsPerImage = patchX_axis*patchY_axis;
    int numberPatchs = numberPatchsPerImage*directoryManager->nfiles;
    //FeatureMatrix* featureMatrix = createFeatureMatrix();
    destroyImage(&firstImage);
    
    int k;
    int dim = dictionary->nFeaturesVectors;
    double r;
    Image* patch;
    FeatureVector* patch_hist = NULL;
    int word_index;
    
    FeatureMatrix* featureMatrix = createFeatureMatrix(directoryManager->nfiles);

// #pragma omp parallel for
    for (size_t fileIndex = 0; fileIndex < directoryManager->nfiles; ++fileIndex) {
        Image* currentImage = readImage(directoryManager->files[fileIndex]->path);
        featureMatrix->featureVector[fileIndex] = createFeatureVector(dim);
        for(int col=0; col<dim; col++){
            featureMatrix->featureVector[fileIndex]->features[col] = 0;
        }
        k = 0;
        for (int y = 0; y <= currentImage->ny-patchSize; y +=patchSize) {
            for (int x = 0; x <= currentImage->nx-patchSize; x += patchSize) {
                patch = extractSubImage(currentImage,x,y,patchSize,patchSize,true);
                if(patch_hist) destroyFeatureVector(&patch_hist);
                patch_hist = computeHistogramForFeatureVector(patch,binSize,true);
                word_index = nearest(patch_hist, dictionary);
                featureMatrix->featureVector[fileIndex]->features[word_index]+=1;
                k++;
                destroyImage(&patch);
            }
        }
        for(int col=0; col<dim; col++){
            featureMatrix->featureVector[fileIndex]->features[col] /= k;
        }
        destroyImage(&currentImage);
    }
    return featureMatrix;
}


FeatureMatrix* sampleFeatureSoftBoW(DirectoryManager* directoryManager,
                                    FeatureMatrix* dictionary,
                                    FeatureExtractorFn featureExtractor,
                                    int patch_x, int patch_y,
                                    double sampling_factor,
                                    int seed){

    srand(seed);

    FeatureMatrix* featureMatrix = createFeatureMatrix(directoryManager->nfiles);

    int k;
    int dim = dictionary->nFeaturesVectors;
    double r;
    Image* patch;
    FeatureVector* patch_hist = NULL;
    FeatureVector* patch_vbow = NULL;

// #pragma omp parallel for
    for (size_t fileIndex = 0; fileIndex < directoryManager->nfiles; ++fileIndex) {
        Image* currentImage = readImage(directoryManager->files[fileIndex]->path);
        featureMatrix->featureVector[fileIndex] = createFeatureVector(dim);
        k = 0;
        for (int y = 0; y <= currentImage->ny - patch_y; ++y) {
            for (int x = 0; x <= currentImage->nx - patch_x; ++x) {
                r = (double)rand() / RAND_MAX; 
                if(r < sampling_factor){
                    patch = extractSubImage(currentImage, x, y,
                                            patch_x, patch_y, true);
                    if(patch_hist) destroyFeatureVector(&patch_hist);
                    if(patch_vbow) destroyFeatureVector(&patch_vbow);
                    patch_hist = featureExtractor(patch);
                    patch_vbow = computeSoftVBoW(patch_hist, dictionary);
                    for(int col=0; col<dim; col++){
                        featureMatrix->featureVector[fileIndex]->features[col] +=
                            patch_vbow->features[col];
                    }
                    k++;
                    destroyImage(&patch);
                }
            }
        }
        for(int col=0; col<dim; col++){
            featureMatrix->featureVector[fileIndex]->features[col] /= k;
        }
        destroyImage(&currentImage);
    }

    return featureMatrix;
}


FeatureMatrix* sampleFeatureBoW(DirectoryManager* directoryManager,
                                FeatureMatrix* dictionary,
                                FeatureExtractorFn featureExtractor,
                                int patch_x, int patch_y,
                                double sampling_factor,
                                int seed){

    srand(seed);

    FeatureMatrix* featureMatrix = createFeatureMatrix(directoryManager->nfiles);

    int k;
    int dim = dictionary->nFeaturesVectors;
    double r;
    Image* patch;
    FeatureVector* patch_hist = NULL;
    int word_index;

// #pragma omp parallel for
    for (size_t fileIndex = 0; fileIndex < directoryManager->nfiles; ++fileIndex) {
        Image* currentImage = readImage(directoryManager->files[fileIndex]->path);
        featureMatrix->featureVector[fileIndex] = createFeatureVector(dim);
        for(int col=0; col<dim; col++){
            featureMatrix->featureVector[fileIndex]->features[col] = 0;
        }
        k = 0;
        for (int y = 0; y <= currentImage->ny - patch_y; ++y) {
            for (int x = 0; x <= currentImage->nx - patch_x; ++x) {
                r = (double)rand() / RAND_MAX; 
                if(r < sampling_factor){
                    patch = extractSubImage(currentImage, x, y,
                                            patch_x, patch_y, true);
                    if(patch_hist) destroyFeatureVector(&patch_hist);
                    patch_hist = featureExtractor(patch);
                    word_index = nearest(patch_hist, dictionary);
                    featureMatrix->featureVector[fileIndex]->features[word_index]+=1;
                    k++;
                    destroyImage(&patch);
                }
            }
        }
        for(int col=0; col<dim; col++){
            featureMatrix->featureVector[fileIndex]->features[col] /= k;
        }
        destroyImage(&currentImage);
    }

    return featureMatrix;
}


FeatureMatrix* computeFeatureVectors(Image* imagePack, int patchSize){
    Image* currentSlice;
    Image* patch;
    Histogram* histogram;
    FeatureVector* patchVector;
    int patchX_axis = imagePack->nx/patchSize;
    int patchY_axis = imagePack->ny/patchSize;
    int numberPatchsPerImage = patchX_axis*patchY_axis;
    int numberPatchs = numberPatchsPerImage*imagePack->nz;
    int binSize = 64;

    FeatureMatrix* featureMatrix = createFeatureMatrix(numberPatchs);
    int k=0;
    for (int z = 0; z < imagePack->nz; ++z) {
        currentSlice = getSlice(imagePack,z);
        for (int y = 0; y <= imagePack->ny-patchSize; y +=patchSize) {
            for (int x = 0; x <= imagePack->nx-patchSize; x += patchSize) {
                patch = extractSubImage(currentSlice,x,y,patchSize,patchSize,true);
                histogram = computeHistogram(patch,binSize,true);
                patchVector = createFeatureVector(histogram);
                featureMatrix->featureVector[k] = patchVector;
                k++;
                destroyHistogram(&histogram);
                destroyImage(&patch);
            }
        }
        destroyImage(&currentSlice);
    }
    return featureMatrix;
}



FeatureMatrix* kMeansClustering(FeatureMatrix* featureMatrix,
                                int numberOfCluster,
                                float* loss,
                                int numIter,
                                int batch_size
                               ){
    FeatureMatrix* dict = createFeatureMatrix(numberOfCluster);
    int i=0, j=0, k=0;
    int num_points = featureMatrix->nFeaturesVectors;
    int randomIndex;
    int dim = featureMatrix->featureVector[0]->size;
    bool *isUsed = (bool*)calloc(num_points,sizeof(*isUsed));
    int* labels = (int*)calloc(num_points,sizeof(*labels));
    int* counts = (int*)calloc(numberOfCluster, sizeof(*counts));
    while (k < numberOfCluster) {
        randomIndex = RandomInteger(0, num_points-1);
        if(isUsed[randomIndex] == false){
            dict->featureVector[k] = copyFeatureVector(featureMatrix->featureVector[randomIndex]);
            isUsed[randomIndex] = true;
            k++;
        }
    }
    free(isUsed);
    
    float d, di;
    double r;
    
    // BEGIN Mini-batch variables
    if(batch_size == 0 || batch_size > num_points){
        batch_size = num_points;
    }
    int b[batch_size]; // Double the sampling factor for security
    int bi, rand_start, bpos;
    double sampling_factor = (double)batch_size / num_points * 2;
    // END Mini-batch variables
    
    for(int iter=0; iter < numIter; iter++){
        // Sampling (batch creation)
        if(batch_size == num_points){
            // In this case, the batch is the full point set
            for(i=0; i<batch_size; i++){
                b[i] = i;
            }
            std::cout << "here" << std::endl;
        }
        else{
            // Some tricks for proper randomization. Point colision exists, but is minimized
            bi = 0;
            rand_start = (int) ( (double)rand() * num_points / RAND_MAX );
            while(bi < batch_size){
                for(i=0; i<num_points; i++){
                    bpos = i + rand_start;
                    if(bpos >= num_points) bpos -= num_points;
                    r = (double)rand() / RAND_MAX; 
                    if(r < sampling_factor){
                        b[bi] = bpos;
                        bi++;
                        if(bi >= batch_size) break;
                    }
                }
            }
        }
        
        // Maximization
        *loss = 0;
        d = std::numeric_limits<float>::max();
        for(i=0; i<batch_size; i++){
            for(j=0; j<numberOfCluster; j++){
                di = vectorManhattanDistance(featureMatrix->featureVector[b[i]],
                                             dict->featureVector[j]);
                if(di < d){
                    d = di;
                    labels[i] = j;
                }
            }
            *loss += d;
        }
        // Expectation
        for(j=0; j<numberOfCluster; j++){
            counts[j] = 0; // Zeroing counts
            for(k=0; k<dim; k++){
                dict->featureVector[j]->features[k] = 0; // Reinitializing centroids
            }
        }
        for(i=0; i<batch_size; i++){
            j = labels[b[i]];
            counts[j]++;
            // Summing all points from same clusters
            for(k=0; k<dim; k++){
                dict->featureVector[j]->features[k] +=
                    featureMatrix->featureVector[b[i]]->features[k];
            }
        }
        for(j=0; j<numberOfCluster; j++){
            if(counts[j] == 0){ // If empty cluster, assign random point as centroid
                randomIndex = RandomInteger(0, featureMatrix->nFeaturesVectors-1);
                destroyFeatureVector(&(dict->featureVector[j]));
                dict->featureVector[j] = copyFeatureVector(featureMatrix->featureVector[randomIndex]);
            }
            else{
                // Averaging points by counts
                for(k=0; k<dim; k++){
                    dict->featureVector[j]->features[k] /= counts[j];
                }
            }
        }
    }
    return dict;
}


FeatureVector* computeSoftVBoW(FeatureVector* fv, FeatureMatrix* dict){
    if(fv == NULL || dict == NULL){
        throw std::runtime_error("NULL arg error!\n");
    }
    if(fv->size != dict->featureVector[0]->size){
        throw std::runtime_error("Trying to extract vbow of vector from dict of different dim!\n");
    }
    FeatureVector* vbow = createFeatureVector(dict->nFeaturesVectors);
    for(int i=0; i<dict->nFeaturesVectors; i++){
        vbow->features[i] = vectorManhattanDistance(fv, dict->featureVector[i]);
    }
    return vbow;
}


FeatureVector* computeHardVBoW(FeatureVector* fv, FeatureMatrix* dict, float th){
    if(fv == NULL || dict == NULL){
        throw std::runtime_error("NULL arg error!\n");
    }
    if(fv->size != dict->featureVector[0]->size){
        throw std::runtime_error("Trying to extract vbow of vector from dict of different dim!\n");
    }
    FeatureVector* vbow = createFeatureVector(dict->nFeaturesVectors);
    float diff;
    for(int i=0; i<dict->nFeaturesVectors; i++){
        diff = vectorManhattanDistance(fv, dict->featureVector[i]);
        if(diff < th){
            vbow->features[i] = 1;
        }
        else{
            vbow->features[i] = 0;
        }
    }
    return vbow;
}
