//
// Created by deangeli on 3/27/17.
//
#include "featureVector.h"
#include <stdexcept>


FeatureVector* createFeatureVector(int size){
    FeatureVector* featureVector = (FeatureVector*)calloc(1,sizeof(FeatureVector));
    featureVector->size = size;
    featureVector->features = (float*)calloc((size_t)size, sizeof(float));
    return featureVector;
}

void destroyFeatureVector(FeatureVector** featureVector){
    free((*featureVector)->features);
    free((*featureVector));
    (*featureVector) = NULL;
}

FeatureVector* createRandomNormalizedFeatureVector(int size){
    FeatureVector* featureVector = (FeatureVector*)calloc(1,sizeof(FeatureVector));
    featureVector->size = size;
    featureVector->features = (float*)calloc((size_t)size, sizeof(float));
    for (int i = 0; i < size; ++i) {
        featureVector->features[i] = randomNormalized();
    }
    return featureVector;
}

void printFeatureVector(FeatureVector* featureVector){
    if(featureVector == NULL){
        printf("FeatureVector pointer is NULL\n");
        return;
    }
    for (int i = 0; i < featureVector->size; ++i) {
        printf("%f ",featureVector->features[i]);
    }
    printf("\n");
}

FeatureVector* mergeFeatureVectors(FeatureVector* vector1,FeatureVector* vector2){
    FeatureVector *mergedVector = NULL;
    if(vector1 == NULL || vector2 == NULL){
        printf("vector1 or/and vector2 are NULL\n");
        return mergedVector;
    }
    int mergedSize = vector1->size + vector2->size;
    mergedVector = createFeatureVector(mergedSize);
    for (int i = 0; i < vector1->size; ++i) {
        mergedVector->features[i] = vector1->features[i];
    }

    for (int i = 0; i < vector2->size; ++i) {
        mergedVector->features[i+vector1->size] = vector2->features[i];
    }
    return mergedVector;
}

FeatureVector* createFeatureVector(float* vec,int size){
    FeatureVector* featureVector = (FeatureVector*)calloc(1,sizeof(FeatureVector));
    featureVector->size = size;
    featureVector->features = (float*)calloc((size_t)size,sizeof(float));
    for (int i = 0; i < size; ++i) {
        featureVector->features[i] = vec[i];
    }
    return featureVector;
}


void wirteFeatureVectors(FeatureVector** vectors, int nVectors, char *filename){
    FILE *fp = fopen(filename,"w");
    for (int i = 0; i < nVectors; ++i) {
        FeatureVector* vec = vectors[i];
        for (int j = 0; j < vec->size; ++j) {
            fprintf(fp,"%f",vec->features[j]);
            if(!(j == vec->size-1)){
                fprintf(fp," ");
            }
        }
        fprintf(fp,"\n");
    }
}


float vectorManhattanDistance(FeatureVector* vector1,FeatureVector* vector2){
    if(vector1->size != vector2->size){
        throw std::runtime_error("vectors mismatch dimensions\n");
    }
    float difference = 0;
    FeatureVector* vector = createFeatureVector(vector1->size);
    float diff;
    for (int i = 0; i < vector1->size; ++i) {
        diff = (vector1->features[i]-vector2->features[i]);
        if(diff < 0){
            diff *= -1;
        }
        vector->features[i] = diff;
        difference += vector->features[i];
    }
    destroyFeatureVector(&vector);
    return difference;
}


float vectorEuclideanDistance(FeatureVector* vector1,FeatureVector* vector2){
    if(vector1->size != vector2->size){
        throw std::runtime_error("vectors mismatch dimensions\n");
    }
    float difference = 0;
    float diff;
    for (int i = 0; i < vector1->size; ++i) {
        diff = (vector1->features[i] - vector2->features[i]);
        diff *= diff;
        difference += diff;
    }
    return sqrt(difference);
}


float vectorCosineDistance(FeatureVector* vector1,FeatureVector* vector2){
    if(vector1->size != vector2->size){
        throw std::runtime_error("vectors mismatch dimensions\n");
    }
    float difference = 0;
    float prod, norm1 = 0, norm2 = 0;
    for (int i = 0; i < vector1->size; ++i) {
        prod = vector1->features[i] * vector2->features[i];
        norm1 += vector1->features[i] * vector1->features[i];
        norm2 += vector2->features[i] * vector2->features[i];
        difference += (vector1->features[i] * vector2->features[i]);
    }
    difference /= sqrt(norm1 * norm2);
    return 1 - (difference);
}


FeatureVector* copyFeatureVector(FeatureVector* featureVector){
    return createFeatureVector(featureVector->features, featureVector->size);
}


FeatureMatrix* createFeatureMatrix(){
    FeatureMatrix* featureMatrix = NULL;
    featureMatrix = (FeatureMatrix*)calloc(1,sizeof(FeatureMatrix));
    featureMatrix->nFeaturesVectors = 0;
    featureMatrix->featureVector = NULL;
    return featureMatrix;
}

FeatureMatrix* createFeatureMatrix(int nFeaturesVectors){
    FeatureMatrix* featureMatrix = NULL;
    featureMatrix = (FeatureMatrix*)calloc(1,sizeof(FeatureMatrix));
    featureMatrix->nFeaturesVectors = nFeaturesVectors;
    featureMatrix->featureVector = (FeatureVector**)calloc((size_t)nFeaturesVectors,sizeof(FeatureVector*));
    return featureMatrix;
}

FeatureMatrix* createFeatureMatrix(int nFeaturesVectors,int vectorSize){
    FeatureMatrix* featureMatrix = NULL;
    featureMatrix = (FeatureMatrix*)calloc(1,sizeof(FeatureMatrix));
    featureMatrix->nFeaturesVectors = nFeaturesVectors;
    featureMatrix->featureVector = (FeatureVector**)calloc((size_t)nFeaturesVectors,sizeof(FeatureVector*));
    for (int i = 0; i < vectorSize; ++i) {
        featureMatrix->featureVector[i] = createFeatureVector(vectorSize);
    }
    return featureMatrix;
}

FeatureMatrix* concatFeatureMatrices(FeatureMatrix* featureMatrix1, FeatureMatrix* featureMatrix2){
    if(featureMatrix1 == NULL || featureMatrix2 == NULL){
        throw std::runtime_error("Trying to concat a null feature matrix!\n");
    }
    if(featureMatrix1->featureVector[0]->size !=
       featureMatrix2->featureVector[0]->size){
        throw std::runtime_error("Trying to concat feature matrices of different dim!\n");
    }
    int n_egs = featureMatrix1->nFeaturesVectors + featureMatrix2->nFeaturesVectors;
    FeatureMatrix *featureMatrix = NULL;
    featureMatrix = (FeatureMatrix*)calloc(1,sizeof(FeatureMatrix));
    featureMatrix->nFeaturesVectors = n_egs;
    featureMatrix->featureVector = (FeatureVector**)calloc((size_t)n_egs,sizeof(FeatureVector*));
    for (int i = 0; i < featureMatrix1->nFeaturesVectors; ++i) {
        featureMatrix->featureVector[i] =
            copyFeatureVector(featureMatrix1->featureVector[i]);
    }
    for (int i = 0; i < featureMatrix2->nFeaturesVectors; ++i) {
        featureMatrix->featureVector[i + featureMatrix1->nFeaturesVectors] =
            copyFeatureVector(featureMatrix2->featureVector[i]);
    }
    return featureMatrix;
}

void destroyFeatureMatrix(FeatureMatrix** featureMatrix){
    if(*featureMatrix == NULL){
        throw std::runtime_error("Feature matrix not allocated!\n");
    }
    for (int i = 0; i < (*featureMatrix)->nFeaturesVectors; ++i) {
        destroyFeatureVector( &((*featureMatrix)->featureVector[i]) );
    }
    free((*featureMatrix)->featureVector);
    free((*featureMatrix));
    *featureMatrix = NULL;
}

void writeFeatureMatrix(FeatureMatrix* featureMatrix, char *filename){
    FILE *fp = fopen(filename,"w");
    for (int i = 0; i < featureMatrix->nFeaturesVectors; ++i) {
        FeatureVector* vec = featureMatrix->featureVector[i];
        for (int j = 0; j < vec->size; ++j) {
            fprintf(fp,"%f",vec->features[j]);
            if(!(j == vec->size-1)){
                fprintf(fp," ");
            }
        }
        fprintf(fp,"\n");
    }
}

void addNewLines(FeatureMatrix** featureMatrix, int numberNewLines){
    FeatureMatrix* aux = *featureMatrix;

//    int numberLines = aux->nFeaturesVectors+numberNewLines;
//    FeatureVector** copy = (FeatureVector**)calloc(numberLines,sizeof(FeatureVector*));
//    for (int i = 0; i < aux->nFeaturesVectors; ++i) {
//        copy[i] = aux->featureVector[i];
//    }
//    free(aux->featureVector);
//    aux->featureVector = copy;
//    aux->nFeaturesVectors = numberLines;

    int numberLines = aux->nFeaturesVectors+numberNewLines;
    FeatureVector** newRows = (FeatureVector**)realloc(aux->featureVector,(numberLines)*sizeof(FeatureVector*));
    aux->featureVector = newRows;
    //aux->featureVector = (FeatureVector**)realloc(aux->featureVector,(numberLines)*sizeof(FeatureVector*));
    aux->nFeaturesVectors = numberLines;
}

void printFeatureMatrix(FeatureMatrix* featureMatrix){
    if(featureMatrix == NULL){
        printf("FeatureMatrix pointer is NULL\n");
        return;
    }
    for (int i = 0; i < featureMatrix->nFeaturesVectors; ++i) {
        printFeatureVector(featureMatrix->featureVector[i]);
    }
    printf("\n");
}


//void sortAt(FeatureVector featureVector, int lastIndex){
//
//    for (int i = lastIndex-1; i >= 0 ; --i) {
//        if(featureVector.features[lastIndex] > featureVector.features[i]){
//            break;
//        }
//    }
//    //featureVector
//}
