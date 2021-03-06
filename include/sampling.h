//
// Created by deangeli on 5/20/17.
//

#ifndef _SAMPLING_H
#define _SAMPLING_H

#include "vector.h"
#include "image.h"

GVector* gridSampling(Image* image, size_t patchSizeX,size_t patchSizeY);
GVector* randomSampling(Image* image, size_t patchSizeX, size_t patchSizeY, size_t nPatchs);
GVector* iftSampling(Image* image, size_t patchSizeX, size_t patchSizeY);

#endif //LIBFL_SAMPLING_H
