//
// Created by deangeli on 4/7/17.
//
#include <cmath>
#include <iostream>
#include "bagOfVisualWords.h"
#include "matrixUtil.h"

/* Assumes all arguments are preallocated */
void computeMVN(Matrix* features, float* means, float* variances){
    // Running means and variances via Welford algorithm
    for(size_t j=0; j<features->numberColumns; ++j){
        means[j] = MATRIX_GET_ELEMENT_PO_AS(float, features, 0, j);  
        variances[j] = 0;
    }
    for(size_t i=1; i<features->numberRows; ++i){
        for(size_t j=0; j<features->numberColumns; ++j){
            float current = MATRIX_GET_ELEMENT_PO_AS(float, features, i, j);
            float old_mean = means[j];
            means[j] += (current - means[j]) / i;
            variances[j] += (current - old_mean) * (current - means[j]);
        }
    }
    for(size_t j=0; j<features->numberColumns; ++j){
        variances[j] /= (features->numberRows - 1);
    }
}

/* Modifies matrix in place, assumes all arguments are preallocated */
void applyMVN(Matrix* features, float* means, float* variances){
    for(size_t i=1; i<features->numberRows; ++i){
        for(size_t j=0; j<features->numberColumns; ++j){
            MATRIX_GET_ELEMENT_PO_AS(float, features, i, j) -= means[j];
            if(variances[j] != 0)
                MATRIX_GET_ELEMENT_PO_AS(float, features, i, j) /= sqrt(variances[j]);
        }
    }
}


BagOfVisualWordsManager* createBagOfVisualWordsManager(){
    BagOfVisualWordsManager* bowManager = (BagOfVisualWordsManager*)calloc(1,sizeof(BagOfVisualWordsManager));
    bowManager->pathsToImages_dictionary = NULL;
    bowManager->pathsToImages_train = NULL;
    bowManager->pathsToImages_test = NULL;
    bowManager->dictionary = NULL;
    bowManager->featureExtractorFunction = NULL;
    bowManager->imageSamplerFunction = NULL;
    bowManager->clusteringFunction = NULL;
    bowManager->argumentListOfFeatureExtractor =NULL;
    bowManager->argumentListOfSampler = NULL;
    bowManager->argumentListOfClustering = NULL;
    bowManager->argumentListOfDistanceFunction = NULL;
    bowManager->freeFunction2SamplerOutput = NULL;
    bowManager->classifier = NULL;
    bowManager->freeFunctionClassifier = NULL;
    bowManager->histogramsTraining = NULL;
    bowManager->labelsTraining = NULL;
    bowManager->means = NULL;
    bowManager->variances = NULL;
    bowManager->storeTrainData = false;
    bowManager->storePredictedData = false;
    return bowManager;
}

void destroyBagOfVisualWordsManager(BagOfVisualWordsManager** pBagOfVisualWordsManager){
    BagOfVisualWordsManager* aux = *pBagOfVisualWordsManager;
    if(aux == NULL){
        return;
    }

    destroyVector(&(aux->pathsToImages_dictionary));
    destroyVector(&(aux->pathsToImages_train));
    destroyVector(&(aux->pathsToImages_test));
    destroyVector(&(aux->labelsTraining));
    destroyVector(&(aux->labelsPredicted));

    destroyArgumentList(&(aux->argumentListOfSampler));
    destroyArgumentList(&(aux->argumentListOfFeatureExtractor));
    destroyArgumentList(&(aux->argumentListOfClustering));
    destroyArgumentList(&(aux->argumentListOfDistanceFunction));
    destroyArgumentList(&(aux->argumentListOfHistogramMounter));

    destroyMatrix(&(aux->dictionary));
    destroyMatrix(&(aux->histogramsTraining));
    destroyMatrix(&(aux->histogramsPredictSamples));
    
    if(aux->means != NULL) free(aux->means);
    if(aux->variances) free(aux->variances);


    if(aux->freeFunctionClassifier){
        if(aux->classifier){
            aux->freeFunctionClassifier(aux->classifier);
        }
    }


    free(*pBagOfVisualWordsManager);
}

//void setDefaultParameters(BagOfVisualWordsManager* bowManager){
//    if(bowManager == NULL){
//        printf("[setDefaultParameters] bowManager is null");
//        return;
//    }
//
//    ArgumentList* gridSamplingArguments = createArgumentList();
//    ARGLIST_PUSH_BACK_AS(size_t,gridSamplingArguments,64); //patch size X
//    ARGLIST_PUSH_BACK_AS(size_t,gridSamplingArguments,64); //patch size Y
//    bowManager->argumentListOfSampler = gridSamplingArguments;
//    bowManager->imageSamplerFunction = gridSampling;
//
//    //Neste exemplo eu usar o histograma de cores
//    ArgumentList* colorFeatureExtractorArguments = createArgumentList();
//    ARGLIST_PUSH_BACK_AS(size_t,colorFeatureExtractorArguments,8); //nBins per channel
//    ARGLIST_PUSH_BACK_AS(size_t,colorFeatureExtractorArguments,8*8*8); //total number of channels
//    bowManager->featureExtractorFunction = computeColorHistogram;
//
//}

GVector* gridSamplingBow(Image* image, BagOfVisualWordsManager* bagOfVisualWordsManager){
    ArgumentList* argumentList = bagOfVisualWordsManager->argumentListOfSampler;

    if(argumentList->length == 0){
        printf("[gridSampling] invalid argument list");
    }else if(argumentList->length == 1){
        size_t patchSize = ARGLIST_GET_ELEMENT_AS(size_t ,argumentList,0);
        return gridSampling(image,patchSize,patchSize);
    }else if(argumentList->length == 2){
        size_t patchSizeX = ARGLIST_GET_ELEMENT_AS(size_t,argumentList,0);
        size_t patchSizeY = ARGLIST_GET_ELEMENT_AS(size_t,argumentList,1);
        return gridSampling(image,patchSizeX,patchSizeY);
    }else{
        return NULL;
    }
    return NULL;
}

GVector* randomSamplingBow(Image* image, BagOfVisualWordsManager* bagOfVisualWordsManager){
    ArgumentList* argumentList = bagOfVisualWordsManager->argumentListOfSampler;

    if(argumentList->length != 3){
        printf("[randomSampling] invalid argument list");
        return NULL;
    }
    size_t patchSizeX = ARGLIST_GET_ELEMENT_AS(size_t,argumentList,0);
    size_t patchSizeY = ARGLIST_GET_ELEMENT_AS(size_t,argumentList,1);
    size_t nPatches = ARGLIST_GET_ELEMENT_AS(size_t,argumentList,2);
    return randomSampling(image,patchSizeX,patchSizeY,nPatches);
}


GVector* iftSamplingBow(Image* image, BagOfVisualWordsManager* bagOfVisualWordsManager){
    ArgumentList* argumentList = bagOfVisualWordsManager->argumentListOfSampler;
    if(argumentList->length != 2){
        printf("[randomSampling] invalid argument list");
        return NULL;
    }
    size_t patchSizeX = ARGLIST_GET_ELEMENT_AS(size_t,argumentList,0);
    size_t patchSizeY = ARGLIST_GET_ELEMENT_AS(size_t,argumentList,1);
    return iftSampling(image,patchSizeX,patchSizeY);
}



Matrix* computeColorHistogramBow(GVector* vector,BagOfVisualWordsManager* bagOfVisualWordsManager){
    ArgumentList* argumentList = bagOfVisualWordsManager->argumentListOfFeatureExtractor;
    if(argumentList->length < 2){
        printf("[computeColorHistogram] invalid argument list");
        return NULL;
    }
    if(vector->size == 0){
        printf("[computeColorHistogram] vector has 0 elements");
        return NULL;
    }
    size_t nbinsPerChannel = ARGLIST_GET_ELEMENT_AS(size_t,argumentList,0);
    size_t totalBins = ARGLIST_GET_ELEMENT_AS(size_t,argumentList,1);
    return computeColorHistogram(vector,nbinsPerChannel,totalBins);
}

Matrix* computeHOGBow(GVector* vector,BagOfVisualWordsManager* bagOfVisualWordsManager){
    ArgumentList* argumentList = bagOfVisualWordsManager->argumentListOfFeatureExtractor;
    if(argumentList->length < 1){
        printf("[computeHOG] invalid argument list");
        return NULL;
    }
    if(vector->size == 0){
        printf("[computeHOG] vector has 0 elements");
        return NULL;
    }
    size_t nbins = ARGLIST_GET_ELEMENT_AS(size_t,argumentList,0);
    return computeHOG(vector, nbins);
}

Matrix* computeHOGPerChannelBow(GVector* vector,BagOfVisualWordsManager* bagOfVisualWordsManager){
    ArgumentList* argumentList = bagOfVisualWordsManager->argumentListOfFeatureExtractor;
    if(argumentList->length < 1){
        printf("[computeHOGPerChannel] invalid argument list");
        return NULL;
    }
    if(vector->size == 0){
        printf("[computeHOGPerChannel] vector has 0 elements");
        return NULL;
    }
    size_t nbins = ARGLIST_GET_ELEMENT_AS(size_t,argumentList,0);
    return computeHOG(vector, nbins, true);
}


Matrix* computeMergedFeats(GVector* vector,BagOfVisualWordsManager* bagOfVisualWordsManager){
    ArgumentList* argumentList = bagOfVisualWordsManager->argumentListOfFeatureExtractor;
    if(argumentList->length != 3){
        printf("[computeMergedFeats] invalid argument list");
        return NULL;
    }
    size_t colorBins = ARGLIST_GET_ELEMENT_AS(size_t,argumentList,0);
    size_t totalColorBins = ARGLIST_GET_ELEMENT_AS(size_t,argumentList,1);
    size_t gradientBins = ARGLIST_GET_ELEMENT_AS(size_t,argumentList,2);
    Matrix* colorFeats = computeColorHistogram(vector,colorBins,totalColorBins);
    Matrix* hogFeats = computeHOG(vector, gradientBins);
    Matrix* feats = stackHorizontallyMatrices(colorFeats, hogFeats);
    destroyMatrix(&colorFeats);
    destroyMatrix(&hogFeats);
    return feats;
}


void computeDictionary(BagOfVisualWordsManager* bagOfVisualWordsManager){
    Matrix* allFeatures = NULL;
    if(!bagOfVisualWordsManager->imageSamplerFunction){
        printf("[computeDictionary] Sampler function not defined\n");
    }

    if(!bagOfVisualWordsManager->featureExtractorFunction){
        printf("[computeDictionary] Feature extractor function not defined\n");
        return;
    }
    if(!bagOfVisualWordsManager->clusteringFunction){
        printf("[computeDictionary] Clustering function not defined\n");
        return;
    }

    printf("[computeDictionary] Generating visual words...\n");
    for (size_t i = 0; i < bagOfVisualWordsManager->pathsToImages_dictionary->size; ++i) {
        char* imagePath = VECTOR_GET_ELEMENT_AS(char*,bagOfVisualWordsManager->pathsToImages_dictionary,i);
        std::cout << imagePath << std::endl;
        Image* image = readImage(imagePath);
        if(image == NULL){
            printf("[computeDictionary] invalid image path: %s",imagePath);
            continue;
        }
        GVector* samplingResults = NULL;
        if(bagOfVisualWordsManager->imageSamplerFunction){
            samplingResults = bagOfVisualWordsManager->imageSamplerFunction(image,
                                                                            bagOfVisualWordsManager);
            samplingResults->freeFunction = bagOfVisualWordsManager->freeFunction2SamplerOutput;
        }else{
            samplingResults = createNullVector(1,sizeof(Image*));
            VECTOR_GET_ELEMENT_AS(Image*,samplingResults,0) = image;
        }
        Matrix* featureMatrix = bagOfVisualWordsManager->featureExtractorFunction(samplingResults,
                                                                                  bagOfVisualWordsManager);
        Matrix* newData = stackVerticallyMatrices(allFeatures,featureMatrix);
        destroyMatrix(&allFeatures);
        allFeatures = newData;
        destroyImage(&image);
        destroyVector(&samplingResults);
        destroyMatrix(&featureMatrix);
    }

    printf("[meanVarianceNorm] Normalizing all features...\n");
    bagOfVisualWordsManager->means = (float*)malloc(sizeof(float) * allFeatures->numberColumns);
    bagOfVisualWordsManager->variances = (float*)malloc(sizeof(float) * allFeatures->numberColumns);

    computeMVN(allFeatures,
               bagOfVisualWordsManager->means,
               bagOfVisualWordsManager->variances);
    applyMVN(allFeatures,
             bagOfVisualWordsManager->means,
             bagOfVisualWordsManager->variances);

    printf("[computeDictionary] Finding Visual words...\n");
    bagOfVisualWordsManager->dictionary = bagOfVisualWordsManager->clusteringFunction(allFeatures,
                                                bagOfVisualWordsManager);
    destroyMatrix(&allFeatures);
    printf("[computeDictionary] Dictionary computed\n");
}

void trainClassifier(BagOfVisualWordsManager* bagOfVisualWordsManager){
    if(!bagOfVisualWordsManager->imageSamplerFunction){
        printf("[trainClassifier] Sampler function not defined\n");
    }


    if(!bagOfVisualWordsManager->featureExtractorFunction){
        printf("[trainClassifier] Feature extractor function not defined\n");
        return;
    }
    if(bagOfVisualWordsManager->dictionary == NULL){
        printf("[trainClassifier] Dictionary is empty\n");
        return;
    }
    if(bagOfVisualWordsManager->mountHistogramFunction == NULL){
        printf("[trainClassifier] Mounter histogram function not defined\n");
        return;
    }
    if(bagOfVisualWordsManager->fitFunction == NULL){
        printf("[trainClassifier] Fit function not defined\n");
        return;
    }


    Matrix *bowHistograms = createMatrix(bagOfVisualWordsManager->pathsToImages_train->size,
                                         bagOfVisualWordsManager->dictionary->numberRows,
                                         sizeof(float));
    GVector *imagesLabels = createNullVector(bagOfVisualWordsManager->pathsToImages_train->size,sizeof(int));
    //Matrix *bowHistograms = NULL;
    printf("[trainClassifier] Generating histograms and labels from images\n");
    for (size_t index = 0; index < bagOfVisualWordsManager->pathsToImages_train->size; ++index) {
        char* imagePath = VECTOR_GET_ELEMENT_AS(char*,bagOfVisualWordsManager->pathsToImages_train,index);
        Image* image = readImage(imagePath);
        if(image == NULL){
            printf("[computeDictionary] Invalid image path: %s",imagePath);
            continue;
        }
        GVector* samplingResults = NULL;
        if(bagOfVisualWordsManager->imageSamplerFunction){
            samplingResults = bagOfVisualWordsManager->imageSamplerFunction(image,
                                                                            bagOfVisualWordsManager);
            samplingResults->freeFunction = bagOfVisualWordsManager->freeFunction2SamplerOutput;
        }else{
            samplingResults = createNullVector(1,sizeof(Image*));
            VECTOR_GET_ELEMENT_AS(Image*,samplingResults,0) = image;
        }
        Matrix* featureMatrix = bagOfVisualWordsManager->featureExtractorFunction(samplingResults, bagOfVisualWordsManager);
        applyMVN(featureMatrix,
                 bagOfVisualWordsManager->means,
                 bagOfVisualWordsManager->variances);


        GVector* histogram = bagOfVisualWordsManager->mountHistogramFunction(featureMatrix,bagOfVisualWordsManager);
        setRowValueGivenVector(bowHistograms,histogram,index);
        VECTOR_GET_ELEMENT_AS(int,imagesLabels,index) = findTrueLabelInName(imagePath);

        destroyImage(&image);
        destroyMatrix(&featureMatrix);
        destroyVector(&histogram);
        destroyVector(&samplingResults);
    }

    printf("[trainClassifier] Histograms and labels generated\n");
    printf("[trainClassifier] Classifier  trainning...\n");

    bagOfVisualWordsManager->fitFunction(bowHistograms,imagesLabels,bagOfVisualWordsManager->classifier);
    printf("[trainClassifier] Classifier trained...\n");

    if(bagOfVisualWordsManager->storeTrainData){
        if(bagOfVisualWordsManager->histogramsTraining){
            destroyMatrix(&(bagOfVisualWordsManager->histogramsTraining));
        }
        bagOfVisualWordsManager->histogramsTraining = bowHistograms;

        if(bagOfVisualWordsManager->labelsTraining){
            destroyVector(&(bagOfVisualWordsManager->labelsTraining));
        }
        bagOfVisualWordsManager->labelsTraining = imagesLabels;
    }else{
        destroyMatrix(&bowHistograms);
        destroyVector(&imagesLabels);
    }
}

GVector* predictLabels(BagOfVisualWordsManager* bagOfVisualWordsManager){
    if(!bagOfVisualWordsManager->imageSamplerFunction){
        printf("[predictLabels] Sampler function not defined\n");
    }


    if(!bagOfVisualWordsManager->featureExtractorFunction){
        printf("[predictLabels] Feature extractor function not defined\n");
        return NULL;
    }
    if(bagOfVisualWordsManager->dictionary == NULL){
        printf("[predictLabels] Dictionary is empty\n");
        return NULL;
    }
    if(bagOfVisualWordsManager->mountHistogramFunction == NULL){
        printf("[predictLabels] Mounter histogram function not defined\n");
        return NULL;
    }
    if(bagOfVisualWordsManager->predictFunction == NULL){
        printf("[predictLabels] Predict function not defined\n");
        return NULL;
    }
    Matrix *bowHistograms = createMatrix(bagOfVisualWordsManager->pathsToImages_test->size,
                                         bagOfVisualWordsManager->dictionary->numberRows,
                                         sizeof(float));
    printf("[predictLabels] Generating histograms and labels from images\n");
    for (size_t index = 0; index < bagOfVisualWordsManager->pathsToImages_test->size; ++index) {
        char* imagePath = VECTOR_GET_ELEMENT_AS(char*,bagOfVisualWordsManager->pathsToImages_test,index);
        Image* image = readImage(imagePath);
        if(image == NULL){
            printf("[predictLabels] Invalid image path: %s",imagePath);
            continue;
        }
        GVector* samplingResults = NULL;
        if(bagOfVisualWordsManager->imageSamplerFunction){
            samplingResults = bagOfVisualWordsManager->imageSamplerFunction(image,
                                                                            bagOfVisualWordsManager);
            samplingResults->freeFunction = bagOfVisualWordsManager->freeFunction2SamplerOutput;
        }else{
            samplingResults = createNullVector(1,sizeof(Image*));
            VECTOR_GET_ELEMENT_AS(Image*,samplingResults,0) = image;
        }
        Matrix* featureMatrix = bagOfVisualWordsManager->featureExtractorFunction(samplingResults, bagOfVisualWordsManager);

        applyMVN(featureMatrix,
                bagOfVisualWordsManager->means,
                bagOfVisualWordsManager->variances);

        GVector* histogram = bagOfVisualWordsManager->mountHistogramFunction(featureMatrix,bagOfVisualWordsManager);
        setRowValueGivenVector(bowHistograms,histogram,index);

        destroyImage(&image);
        destroyMatrix(&featureMatrix);
        destroyVector(&histogram);
        destroyVector(&samplingResults);
    }
    printf("[predictLabels] Histograms and labels generated\n");
    printf("[predictLabels] Predicting labels...\n");

    GVector* labelsPredicted = bagOfVisualWordsManager->predictFunction(bowHistograms,bagOfVisualWordsManager->classifier);
    printf("[predictLabels] Labels predicted...\n");
    if(bagOfVisualWordsManager->storePredictedData){
        if(bagOfVisualWordsManager->histogramsPredictSamples){
            destroyMatrix(&(bagOfVisualWordsManager->histogramsPredictSamples));
        }
        bagOfVisualWordsManager->histogramsPredictSamples = bowHistograms;
        if(bagOfVisualWordsManager->labelsPredicted){
            destroyVector(&(bagOfVisualWordsManager->labelsPredicted));
        }
        bagOfVisualWordsManager->labelsPredicted = labelsPredicted;
    }else{
        destroyMatrix(&bowHistograms);
    }
    return labelsPredicted;
}


/* Very crude modification. Uses the datasets and classifier from the first manager only.*/
void trainClassifier(std::vector<BagOfVisualWordsManager*> managers){
    size_t numRows = 0;
    if(managers[0]->fitFunction == NULL){
        printf("[trainClassifier] Fit function not defined\n");
        return;
    }
    for(BagOfVisualWordsManager* manager : managers){
        if(!manager->imageSamplerFunction){
            printf("[trainClassifier] Sampler function not defined\n");
        }
        if(!manager->featureExtractorFunction){
            printf("[trainClassifier] Feature extractor function not defined\n");
            return;
        }
        if(manager->dictionary == NULL){
            printf("[trainClassifier] Dictionary is empty\n");
            return;
        }
        if(manager->mountHistogramFunction == NULL){
            printf("[trainClassifier] Mounter histogram function not defined\n");
            return;
        }
        numRows += manager->dictionary->numberRows;
    }

    
    Matrix *bowHistograms = createMatrix(managers[0]->pathsToImages_train->size,
                                         numRows, sizeof(float));
    GVector *imagesLabels = createNullVector(managers[0]->pathsToImages_train->size,sizeof(int));

    printf("[trainClassifier] Generating histograms and labels from images\n");
    for (size_t index = 0; index < managers[0]->pathsToImages_train->size; ++index) {
        char* imagePath = VECTOR_GET_ELEMENT_AS(char*,managers[0]->pathsToImages_train,index);
        Image* image = readImage(imagePath);
        if(image == NULL){
            printf("[computeDictionary] Invalid image path: %s",imagePath);
            continue;
        }
        GVector* samplingResults = NULL;
        size_t colOffset = 0;
        for(BagOfVisualWordsManager* manager : managers){
            if(manager->imageSamplerFunction){
                samplingResults = manager->imageSamplerFunction(image,
                                                                manager);
                samplingResults->freeFunction = manager->freeFunction2SamplerOutput;
            }else{
                samplingResults = createNullVector(1,sizeof(Image*));
                VECTOR_GET_ELEMENT_AS(Image*,samplingResults,0) = image;
            }
            Matrix* featureMatrix = manager->featureExtractorFunction(samplingResults, manager);
            applyMVN(featureMatrix,
                     manager->means,
                     manager->variances);

            GVector* histogram = manager->mountHistogramFunction(featureMatrix,manager);
            for(size_t col=0; col < manager->dictionary->numberRows; col++){
                MATRIX_GET_ELEMENT_PO_AS(float, bowHistograms, index, col + colOffset) = 
                    VECTOR_GET_ELEMENT_AS(float, histogram, col);
            }
            colOffset += manager->dictionary->numberRows;
            destroyMatrix(&featureMatrix);
            destroyVector(&histogram);
        }
        VECTOR_GET_ELEMENT_AS(int,imagesLabels,index) = findTrueLabelInName(imagePath);

        destroyImage(&image);
        destroyVector(&samplingResults);
    }

    printf("[trainClassifier] Histograms and labels generated\n");
    printf("[trainClassifier] Classifier  trainning...\n");
 
    managers[0]->fitFunction(bowHistograms,imagesLabels,managers[0]->classifier);
    printf("[trainClassifier] Classifier trained...\n");

    destroyMatrix(&bowHistograms);
    destroyVector(&imagesLabels);
}


GVector* predictLabels(std::vector<BagOfVisualWordsManager*> managers){
    size_t numRows = 0;
    if(managers[0]->fitFunction == NULL){
        printf("[predictLabels] Fit function not defined\n");
        return NULL;
    }
    for(BagOfVisualWordsManager* manager : managers){
        if(!manager->imageSamplerFunction){
            printf("[predictLabels] Sampler function not defined\n");
        }
        if(!manager->featureExtractorFunction){
            printf("[predictLabels] Feature extractor function not defined\n");
            return NULL;
        }
        if(manager->dictionary == NULL){
            printf("[predictLabels] Dictionary is empty\n");
            return NULL;
        }
        if(manager->mountHistogramFunction == NULL){
            printf("[predictLabels] Mounter histogram function not defined\n");
            return NULL;
        }
        numRows += manager->dictionary->numberRows;
    }

    
    Matrix *bowHistograms = createMatrix(managers[0]->pathsToImages_test->size,
                                         numRows, sizeof(float));
    printf("[predictLabels] Generating histograms and labels from images\n");
    for (size_t index = 0; index < managers[0]->pathsToImages_test->size; ++index) {
        char* imagePath = VECTOR_GET_ELEMENT_AS(char*,managers[0]->pathsToImages_test,index);
        Image* image = readImage(imagePath);
        if(image == NULL){
            printf("[computeDictionary] Invalid image path: %s",imagePath);
            continue;
        }
        GVector* samplingResults = NULL;
        size_t colOffset = 0;
        for(BagOfVisualWordsManager* manager : managers){
            if(manager->imageSamplerFunction){
                samplingResults = manager->imageSamplerFunction(image,
                                                                manager);
                samplingResults->freeFunction = manager->freeFunction2SamplerOutput;
            }else{
                samplingResults = createNullVector(1,sizeof(Image*));
                VECTOR_GET_ELEMENT_AS(Image*,samplingResults,0) = image;
            }
            Matrix* featureMatrix = manager->featureExtractorFunction(samplingResults, manager);
            applyMVN(featureMatrix,
                     manager->means,
                     manager->variances);

            GVector* histogram = manager->mountHistogramFunction(featureMatrix,manager);
            for(size_t col=0; col < manager->dictionary->numberRows; col++){
                MATRIX_GET_ELEMENT_PO_AS(float, bowHistograms, index, col + colOffset) = 
                    VECTOR_GET_ELEMENT_AS(float, histogram, col);
            }
            colOffset += manager->dictionary->numberRows;
            destroyMatrix(&featureMatrix);
            destroyVector(&histogram);
        }
        destroyImage(&image);
        destroyVector(&samplingResults);
    }
    printf("[predictLabels] Histograms and labels generated\n");
    printf("[predictLabels] Predicting labels...\n");

    GVector* labelsPredicted = managers[0]->predictFunction(bowHistograms,managers[0]->classifier);
    printf("[predictLabels] Labels predicted...\n");
    destroyMatrix(&bowHistograms);
    return labelsPredicted;
}


GVector* computeCountHistogram_bow(Matrix* featureMatrix,BagOfVisualWordsManager* bagOfVisualWordsManager){
    GVector* bowHistogram = createNullVector(bagOfVisualWordsManager->dictionary->numberRows,sizeof(float));
    for (size_t patchIndex = 0; patchIndex < featureMatrix->numberRows; ++patchIndex) {
        size_t nearestClusterIndex = findNearestRow(bagOfVisualWordsManager->dictionary,
                                                    featureMatrix,patchIndex,
                                                    bagOfVisualWordsManager->distanceFunction,
                                                    bagOfVisualWordsManager->argumentListOfDistanceFunction);
        VECTOR_GET_ELEMENT_AS(float,bowHistogram,nearestClusterIndex) += 1.0;
    }
    for (size_t i = 0; i < bowHistogram->size; ++i) {
        VECTOR_GET_ELEMENT_AS(float,bowHistogram,i) /= featureMatrix->numberRows;
    }
    return bowHistogram;
}

GVector* computeCountHistogram_softBow(Matrix* featureMatrix,BagOfVisualWordsManager* bagOfVisualWordsManager){
    GVector* bowHistogram = createNullVector(bagOfVisualWordsManager->dictionary->numberRows,sizeof(float));
    double l1norm;
    double dist;
    l1norm = 0;
    for (size_t patchIndex = 0; patchIndex < featureMatrix->numberRows; ++patchIndex) {
        for (size_t wordIndex = 0; wordIndex < bagOfVisualWordsManager->dictionary->numberRows; ++wordIndex){
            dist = computeDistanceBetweenRows(featureMatrix, bagOfVisualWordsManager->dictionary,
                                              patchIndex, wordIndex);
            VECTOR_GET_ELEMENT_AS(float, bowHistogram, wordIndex) += dist;
            l1norm += dist;
        }
    }
    for (size_t i = 0; i < bowHistogram->size; ++i) {
        VECTOR_GET_ELEMENT_AS(float,bowHistogram,i) /= l1norm;
    }
    return bowHistogram;
}



Matrix* kmeansClusteringBow(Matrix* featureMatrix, BagOfVisualWordsManager* bagOfVisualWordsManager){
    ArgumentList* argumentList = bagOfVisualWordsManager->argumentListOfClustering;
    if(argumentList->length <= 0){
        printf("[kmeansClustering] invalid argument list\n");
        return NULL;
    }
    if(featureMatrix->numberRows == 0 || featureMatrix->numberColumns == 0){
        printf("[kmeansClustering] invalid matrix dimension: R%lu C%lu\n",featureMatrix->numberRows,featureMatrix->numberColumns);
        return NULL;
    }

    if(argumentList->length == 1){
        size_t numberOfCluster = ARGLIST_GET_ELEMENT_AS(size_t,argumentList,0);
        return kmeansClustering(featureMatrix,numberOfCluster);
    }
    else if(argumentList->length == 2){
        size_t numberOfCluster = ARGLIST_GET_ELEMENT_AS(size_t,argumentList,0);
        size_t maximumNumberIterations = ARGLIST_GET_ELEMENT_AS(size_t,argumentList,1);
        return kmeansClustering(featureMatrix,numberOfCluster,
                                maximumNumberIterations);
    }else if(argumentList->length == 3){
        size_t numberOfCluster = ARGLIST_GET_ELEMENT_AS(size_t,argumentList,0);
        size_t maximumNumberIterations = ARGLIST_GET_ELEMENT_AS(size_t,argumentList,1);
        double tolerance = ARGLIST_GET_ELEMENT_AS(double,argumentList,2);

        return kmeansClustering(featureMatrix,numberOfCluster,
                                maximumNumberIterations,tolerance);
    }else if(argumentList->length == 4){
        size_t numberOfCluster = ARGLIST_GET_ELEMENT_AS(size_t,argumentList,0);
        size_t maximumNumberIterations = ARGLIST_GET_ELEMENT_AS(size_t,argumentList,1);
        double tolerance = ARGLIST_GET_ELEMENT_AS(double,argumentList,2);
        int seed = ARGLIST_GET_ELEMENT_AS(int,argumentList,3);
        return kmeansClustering(featureMatrix,numberOfCluster,
                                maximumNumberIterations,tolerance,seed);
    }else if (argumentList->length == 6){
        size_t numberOfCluster = ARGLIST_GET_ELEMENT_AS(size_t,argumentList,0);
        size_t maximumNumberIterations = ARGLIST_GET_ELEMENT_AS(size_t,argumentList,1);
        double tolerance = ARGLIST_GET_ELEMENT_AS(double,argumentList,2);
        int seed = ARGLIST_GET_ELEMENT_AS(int,argumentList,3);
        DistanceFunction distanceFunction = ARGLIST_GET_ELEMENT_AS(DistanceFunction ,argumentList,4);
        ArgumentList* distanceFunctionArgs = ARGLIST_GET_ELEMENT_AS(ArgumentList* ,argumentList,5);
        return kmeansClustering(featureMatrix,numberOfCluster,
                                maximumNumberIterations,tolerance,seed,
                                distanceFunction, distanceFunctionArgs);
    }else{
        printf("[kmeansClusteringBow] invalid arguments for kmeans\n");
    }
    return NULL;

}





