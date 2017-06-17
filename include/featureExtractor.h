//
// Created by deangeli on 5/20/17.
//

#ifndef LIBFL_FEATUREEXTRACTOR_H
#define LIBFL_FEATUREEXTRACTOR_H

#include "vector.h"
#include "matrix.h"
#include "image.h"
#include "histogram.h"


Matrix* computeColorHistogram(GVector* vector_images,size_t nbinsPerChannel,size_t totalNumberBins);

Matrix* computeHOG(GVector* vector_images, size_t nbins, bool perChannel=false);



#endif //LIBFL_FEATUREEXTRACTOR_H
