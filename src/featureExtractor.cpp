//
// Created by deangeli on 5/20/17.
//
#include "featureExtractor.h"
#include "hog.h"

Matrix* computeColorHistogram(GVector* vector_images,size_t nbinsPerChannel,size_t totalNumberBins){
    Matrix* matrix = createMatrix(vector_images->size,totalNumberBins,sizeof(float));
    int k = 0;
    for (size_t i = 0; i < vector_images->size; ++i) {
        Image* patch = VECTOR_GET_ELEMENT_AS(Image*,vector_images,i);
        GVector* featureVector = computeHistogramForFeatureVectorGivenNBins(patch, nbinsPerChannel, true);
        for (size_t j = 0; j < matrix->numberColumns; ++j) {
            MATRIX_GET_ELEMENT_BI_AS(float, matrix, k) = VECTOR_GET_ELEMENT_AS(float,featureVector,j);
            k++;
        }
        destroyVector(&featureVector);
    }
    return matrix;
}

Matrix* computeHOG(GVector* vector_images, size_t nbins, bool perChannel){
    Matrix* matrix = createMatrix(vector_images->size, nbins, sizeof(float));
    int k = 0;
    for (size_t i = 0; i < vector_images->size; ++i) {
        Image* patch = VECTOR_GET_ELEMENT_AS(Image*,vector_images,i);
        int sX = (patch->nx - 2) / 2;
        int sY = (patch->ny - 2) / 2;
        int strideX = sX / 2;
        int strideY = sY / 2;
        GVector* featureVector = computeHOGForFeatureVectorGivenNBins(patch, sX, sY, strideX, strideY, nbins, perChannel);
        for (size_t j = 0; j < matrix->numberColumns; ++j) {
            MATRIX_GET_ELEMENT_BI_AS(float, matrix, k) = VECTOR_GET_ELEMENT_AS(float,featureVector,j);
            k++;
        }
        destroyVector(&featureVector);
    }
    return matrix;
}
