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

std::vector<int> knn(FeatureMatrix* target, FeatureMatrix* trainX,
                     std::vector<int> trainY,
                     int k,
                     VectorDistFn distFn=vectorManhattanDistance
                    );

#endif //_KNEARESTNEIGHBOURS_H_