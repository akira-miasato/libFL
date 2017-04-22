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

FeatureMatrix* computeFeatureVectors(Image* imagePack, int patchSize);
FeatureMatrix* sampleHistograms(DirectoryManager* directoryManager,
                                int patch_x, int patch_y,
                                double sampling_factor,
                                int binSize=64, int seed=0);
FeatureMatrix* sampleBoW(DirectoryManager* directoryManager,
                         FeatureMatrix* dictionary,
                         int patch_x, int patch_y,
                         double sampling_factor,
                         int binSize=64, int seed=0);
FeatureMatrix* kMeansClustering(FeatureMatrix* featureMatrix, int numberOfCluster);




#endif //LIBFL_BAGOFVISUALWORDS_H
