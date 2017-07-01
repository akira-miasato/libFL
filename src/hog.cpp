#include "hog.h"

#define _USE_MATH_DEFINES
#include <cmath>
#include <tgmath.h>
#include <iostream>

GVector* computeHOGForFeatureVectorGivenNBins(Image *image, int sX, int sY, int strideX, int strideY, int nbins, bool perChannel) {

    GVector *histVec;

    if(image->ny < sY + 2 || image->ny < sX + 2) {
        printf("WARNING: invalid HOG adjacency");
        return NULL;
    }
    
    int nX = (image->nx - sX) / strideX + 1;
    int nY = (image->ny - sY) / strideY + 1;
//     if(nX < 1) nX = 1;
//     if(nY < 1) nY = 1;
//     
//     std::cout << "nX" << nX << "nY" << nY << std::endl;

    float binSize = M_PI * 2 / nbins;

    if(perChannel) {
        histVec = createNullVector(nbins * nX * nY * image->nchannels, sizeof(float));
    }
    else {
        histVec = createNullVector(nbins * nX * nY, sizeof(float));
    }

    for (int begY = 0; begY < nY; begY++) {
        for (int begX = 0; begX < nX; begX++) {
            float l2norm = 0;
            for (int y = 0; y < sY; y++) {
                int yp = begY * strideY + y + 1;
                for (int x = 0; x < sX; x++) {
                    int xp = begX * strideX + x + 1;
                    int upper = (yp - 1) * image->nx + xp;
                    int lower = (yp + 1) * image->nx + xp;
                    float dh = 0;

                    int left =  yp * image->nx + (xp - 1);
                    int right = yp * image->nx + (xp + 1);
                    float dw = 0;

                    for (int cp = 0; cp < image->nchannels; cp++) {
                        if(perChannel) {
                            dh = image->channel[cp][lower] - image->channel[cp][upper];
                            dw = image->channel[cp][right] - image->channel[cp][left];
                        }
                        else {
                            dh += image->channel[cp][lower] - image->channel[cp][upper];
                            dw += image->channel[cp][right] - image->channel[cp][left];
                        }

                        if(perChannel || cp == image->nchannels - 1) {
                            size_t offset;
                            if(perChannel) offset = nbins*cp;
                            else offset = 0;

                            double mag = dh * dh + dw * dw;
                            if(mag == 0) continue;

                            l2norm += mag;
                            float angle;

                            angle = atan2(dw, dh) + M_PI; // Offsets to values to [0,2*M_PI], doesn't bother with phase shift

                            size_t index1 = floor(nbins * angle / M_PI / 2);
                            size_t index2 = ceil(nbins * angle / M_PI / 2);
                            
                            if(index1 == index2) {
                                if(index1 == nbins) index1 = 0;
                                VECTOR_GET_ELEMENT_AS(float, histVec, index1 + offset) += sqrt(mag);
                                continue;
                            }

                            float ratio1 = 1. - abs(angle - binSize * index1) / binSize;
                            float ratio2 = 1. - abs(angle - binSize * index2) / binSize;
                            float den = ratio1 + ratio2;
                            ratio1 /= den;
                            ratio2 /= den;
                            if(index1 >= nbins) index1 %= nbins;
                            if(index2 >= nbins) index2 %= nbins;

                            VECTOR_GET_ELEMENT_AS(float, histVec, index1 + offset) += sqrt(mag) * ratio1;
                            VECTOR_GET_ELEMENT_AS(float, histVec, index2 + offset) += sqrt(mag) * ratio2;

                        }
                    }
//                     std::cout << xp << "X" << yp << std::endl;
                }
            }
            l2norm = sqrt(l2norm);
            int hoffset = (begY * nX + begX) * histVec->size / (nX*nY);
            if(l2norm != 0) {
                for (size_t i = 0; i < histVec->size / (nX*nY); ++i) {
                    VECTOR_GET_ELEMENT_AS(float, histVec, i + hoffset) /= l2norm;
                }
            }

        }
    }

    return histVec;
}

