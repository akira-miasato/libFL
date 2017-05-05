/*
 *Created by Deangeli Gomes Neves
 *
 * This software may be freely redistributed under the terms
 * of the MIT license.
 *
 */

#ifndef _BAGOFVISUALWORDS_H_
#define _BAGOFVISUALWORDS_H_

#include "common.h"
#include "featureVector.h"
#include "file.h"
#include "image.h"
#include "histogram.h"


typedef struct _bagOfVisualWords {
    int patchSize;
    DirectoryManager* directoryManager;
    FeatureMatrix* vocabulary;
} BagOfVisualWords;


typedef FeatureVector* (*FeatureExtractorFn)(Image* img);

FeatureMatrix* computeHistogramPatches(DirectoryManager* directoryManager, int patchSize);

FeatureMatrix* computeFeatureVectors(Image* imagePack, int patchSize);

FeatureMatrix* sampleFeatures(DirectoryManager* directoryManager,
                              FeatureExtractorFn featureExtractor,
                              int patch_x, int patch_y,
                              double sampling_factor,
                              int seed=0);

FeatureMatrix* patchHistSoftBow(DirectoryManager* directoryManager,
                                FeatureMatrix* dictionary,
                                int patchSize);

FeatureMatrix* patchHistBow(DirectoryManager* directoryManager,
                            FeatureMatrix* dictionary,
                            int patchSize);

FeatureMatrix* sampleFeatureSoftBoW(DirectoryManager* directoryManager,
                                    FeatureMatrix* dictionary,
                                    FeatureExtractorFn featureExtractor,
                                    int patch_x, int patch_y,
                                    double sampling_factor,
                                    int seed=0);

FeatureMatrix* sampleFeatureBoW(DirectoryManager* directoryManager,
                                FeatureMatrix* dictionary,
                                FeatureExtractorFn featureExtractor,
                                int patch_x, int patch_y,
                                double sampling_factor,
                                int seed=0);

FeatureMatrix* kMeansClustering(FeatureMatrix* featureMatrix,
                                int numberOfCluster,
                                float* loss,
                                int numIter=20);

FeatureVector* computeSoftVBoW(FeatureVector* fv, FeatureMatrix* dict);
FeatureVector* computeHardVBoW(FeatureVector* fv, FeatureMatrix* dict, float th);




#endif //LIBFL_BAGOFVISUALWORDS_H
