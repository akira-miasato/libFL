/*
 *Created by Deangeli Gomes Neves
 *
 * This software may be freely redistributed under the terms
 * of the MIT license.
 *
 */

#ifndef _FEATUREVECTOR_H_
#define _FEATUREVECTOR_H_

#include "common.h"

typedef struct _featureVector {
    float* features;
    int size;
} FeatureVector;

/*
 * estrutura para armazenar varios features vectors
 * */
typedef struct _featureMatrix {
    FeatureVector **featureVector;
    int nFeaturesVectors;
} FeatureMatrix;


FeatureVector* createFeatureVector(int size);
FeatureVector* createFeatureVector(float* vec,int size);
FeatureVector* createRandomNormalizedFeatureVector(int size);

/*
 * escreve uma arquivo txt onde caada linha no arquivo e um feature vector
 * */
void wirteFeatureVector(FeatureVector* vector, FILE *fp);
void wirteFeatureVectors(FeatureVector** vectors, int nVectors, char *filename);
void destroyFeatureVector(FeatureVector** vector);
void printFeatureVector(FeatureVector* featureVector);
FeatureVector* mergeFeatureVectors(FeatureVector* vector1,FeatureVector* vector2);
FeatureVector* copyFeatureVector(FeatureVector* featureVector);


/*
 * Funções de união de matrizes de features.
 * Concat unifica por linhas (junta exemplos de mesma dimensão)
 * Merge unifica por colunas (junta features dos mesmos exemplos)
 */

/*Concatena duas matrizes de features, retornando uma terceira*/
FeatureMatrix* concatFeatureMatrices(FeatureMatrix* featureMatrix1, FeatureMatrix* featureMatrix2);

/*
 * Funções de distância
 */
float vectorDifference(FeatureVector* vector1,FeatureVector* vector2); // Minkowsky
float vectorEuclideanDistance(FeatureVector* vector1,FeatureVector* vector2);
float vectorCosineDistance(FeatureVector* vector1,FeatureVector* vector2);

/*
 * escreve uma arquivo txt onde cada linha no arquivo e um feature vector
 * */
FeatureMatrix* createFeatureMatrix();
FeatureMatrix* createFeatureMatrix(int nFeaturesVectors);
FeatureMatrix* createFeatureMatrix(int nFeaturesVectors,int vectorSize);
void addNewLines(FeatureMatrix** featureMatrix, int numberNewLines);
void printFeatureMatrix(FeatureMatrix* featureMatrix);
void writeFeatureMatrix(FeatureMatrix* featureMatrix, char *filename);
void destroyFeatureMatrix(FeatureMatrix** featureMatrix);



void sortAt(FeatureVector featureVector, int lastIndex);

#endif //LIBFL_FEATUREVECTOR_H
