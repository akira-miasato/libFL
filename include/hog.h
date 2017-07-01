#ifndef LIBFL_HOG_H
#define LIBFL_HOG_H

#include "vector.h"
#include "image.h"

GVector* computeHOGForFeatureVectorGivenNBins(Image *image, int sX, int sY, int strideX, int strideY, int nbins, bool perChannel);

#endif
