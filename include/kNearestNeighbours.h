#ifndef _KNEARESTNEIGHBOURS_H_
#define _KNEARESTNEIGHBOURS_H_

#include <vector>
#include <algorithm>
#include <string>
#include <stdexcept>
#include <limits>
#include <utility>

#include "featureVector.h"

typedef  float (*VectorDistFn)(FeatureVector* a, FeatureVector* b);

int nearest(FeatureVector* ref, FeatureMatrix* hyps,
            VectorDistFn distFn=vectorEuclideanDistance);

std::vector<int> knn(FeatureMatrix* target, FeatureMatrix* trainX,
                     std::vector<int> trainY,
                     int k,
                     VectorDistFn distFn=vectorEuclideanDistance);

#endif //_KNEARESTNEIGHBOURS_H_
