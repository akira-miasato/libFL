#ifndef _IFTPOI_H
#define _IFTPOI_H

#include "ift.h"

iftImage *iftCompute_ISF_MIX_MEAN_Superpixels(iftImage *img, int nsuperpixels, float alpha, float beta, int niters, int smooth_niters, int *nseeds, int *finalniters);

#endif //_IFTPOI_H
