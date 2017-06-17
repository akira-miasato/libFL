#include "hog.h"

#define _USE_MATH_DEFINES
#include <cmath>
#include <tgmath.h>

GVector* computeHOGForFeatureVectorGivenNBins(Image *image, int nbins, bool perChannel) {

    GVector *histVec;
    
    if(image->ny < 3 || image->ny < 3){
        printf("WARNING: HoG won't accept adjacencies smaller than 3x3");
        return NULL;
    }
    
    if(perChannel){
        histVec = createNullVector(nbins * image->nchannels, sizeof(float));
    }
    else{
        histVec = createNullVector(nbins, sizeof(float));
    }
    
    float binSize = M_PI / nbins;
    
    float l2norm = 0;

    for (int yp = 1; yp < image->ny - 1; yp++) {
        for (int xp = 1; xp < image->nx - 1; xp++) {
            int upper = (yp - 1) * image->nx + xp;
            int lower = (yp + 1) * image->nx + xp;
            float dh = 0;
            
            int left =  yp * image->nx + (xp - 1);
            int right = yp * image->nx + (xp + 1);
            float dw = 0;

            for (int cp = 0; cp < image->nchannels; cp++) {
                if(perChannel){
                    dh = image->channel[cp][lower] - image->channel[cp][upper];
                    dw = image->channel[cp][right] - image->channel[cp][left];
                }
                else {
                    dh += image->channel[cp][lower] - image->channel[cp][upper];
                    dw += image->channel[cp][right] - image->channel[cp][left];
                }

                if(perChannel || cp == image->nchannels - 1){
                    size_t offset;
                    if(perChannel) offset = nbins*cp;
                    else offset = 0;
                    
                    double mag = dh * dh + dw * dw;
                    if(mag == 0) continue;

                    l2norm += mag;
                    float angle;
                    
                    if(dh == 0) angle = 0;
                    else if(dw == 0) angle = M_PI / 2;
                    else {
                        angle = atan(dh / dw); // Restricts to values from 0 to M_PI
                        if(angle < 0) angle += M_PI;
                    }
                    size_t index1 = floor(nbins * angle / M_PI);
                    size_t index2 = ceil(nbins * angle / M_PI);

                    if(index1 == index2){
                        VECTOR_GET_ELEMENT_AS(float, histVec, index1 + offset) += sqrt(mag);
                        continue;
                    }
                    
                    float ratio1 = 1. - abs(angle - binSize * index1) / binSize;
                    float ratio2 = 1. - abs(angle - binSize * index2) / binSize;
                    float den = ratio1 + ratio2;
                    ratio1 /= den;
                    ratio2 /= den;
                    if(index2 == nbins) index2 = 0;
                    
                    VECTOR_GET_ELEMENT_AS(float, histVec, index1 + offset) += sqrt(mag) * ratio1;
                    VECTOR_GET_ELEMENT_AS(float, histVec, index2 + offset) += sqrt(mag) * ratio2;
                }
            }
        }
    }

    l2norm = sqrt(l2norm);
    if(l2norm != 0) for (size_t i = 0; i < histVec->size; ++i) {
        VECTOR_GET_ELEMENT_AS(float, histVec, i) /= l2norm;
    }

    return histVec;
}
