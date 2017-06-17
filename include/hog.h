#ifndef LIBFL_HOG_H
#define LIBFL_HOG_H

#include "vector.h"
#include "image.h"

GVector* computeHOGForFeatureVectorGivenNBins(Image *image, int nbins, bool perChannel);

#endif
