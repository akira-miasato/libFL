//
// Created by deangeli on 5/20/17.
//
#include <random>
#include <vector>
#include <stdexcept>
#include "sampling.h"
#include "iftPoi.h"

GVector* gridSampling(Image* image, size_t patchSizeX,size_t patchSizeY){
    size_t nPatchs_X = image->nx/patchSizeX;
    size_t nPatchs_Y = image->ny/patchSizeY;
    size_t nPatchs = nPatchs_X*nPatchs_Y;
    GVector* vector_images = createNullVector(nPatchs,sizeof(Image*));
    int k = 0;
    for (size_t y = 0; y <= (size_t)image->ny-patchSizeY; y +=patchSizeY) {
        for (size_t x = 0; x <= (size_t)image->nx-patchSizeX; x += patchSizeX) {
            VECTOR_GET_ELEMENT_AS(Image*,vector_images,k) = extractSubImage(image,x,y,patchSizeX,patchSizeY,true);
            k++;
        }
    }
    return vector_images;
}


GVector* randomSampling(Image* image, size_t patchSizeX, size_t patchSizeY, size_t nPatchs){
    size_t xSlide = image->nx - patchSizeX;
    size_t ySlide = image->ny - patchSizeY;
    float samplingFactor = (float)nPatchs / (ySlide * xSlide);
    size_t offX = (float) rand() * xSlide / RAND_MAX ;
    size_t offY = (float) rand() * ySlide / RAND_MAX ;
    GVector* vector_images = createNullVector(nPatchs,sizeof(Image*));
    size_t k = 0;
    while(k < nPatchs){
        for (size_t y = 0; y <= (size_t)image->ny-patchSizeY; y++) {
            size_t yo = y + offY;
            if(yo > ySlide) yo -= ySlide;
            for (size_t x = 0; x <= (size_t)image->nx-patchSizeX; x++) {
                size_t xo = x + offX;
                if(xo > ySlide) xo -= xSlide;
                if((float) rand() / RAND_MAX < samplingFactor){
                    VECTOR_GET_ELEMENT_AS(Image*,vector_images,k) = extractSubImage(image,xo,yo,patchSizeX,patchSizeY,true);
                    k++;
                    if(k == nPatchs) break;
                }
            }
            if(k == nPatchs) break;
        }
    }
    return vector_images;
}


iftImage* convertImageToIftImage(Image* image){
    iftImage *img;
    if(image->nchannels == 1) img = iftCreateImage(image->nx, image->ny, image->nz);
    else if (image->nchannels == 3) img = iftCreateColorImage(image->nx, image->ny, image->nz);
    else{
        printf("[WARNING] IFT images only support RGB and Grayscale images");
        return NULL;
    }
    for(int  i=0; i< image->nx; i++){
        for(int  j=0; j< image->ny; j++){
            if(image->nchannels == 1){
                iftImgElem2D(img, i, j) = 255 * imageVal(image, i, j);
            }
            else {
                iftImgElem2D(img, i, j) = 255 * imageValCh(image, i, j, 0);
                iftImgCbElem2D(img, i, j) = 255 * imageValCh(image, i, j, 1);
                iftImgCrElem2D(img, i, j) = 255 * imageValCh(image, i, j, 2);
            }
        }
    }
    return img;
}


GVector* iftSampling(Image* image, size_t patchSizeX, size_t patchSizeY){
    iftImage  *img, *label;
    iftImage  *border=NULL;
    int        niters, nseeds;
    int        normvalue;
  
//    "Usage: iftISF_MIX_MEAN <image.[pgm,ppm,png]> <nsamples> <alpha (e.g., [0.005-0.2])> <beta (e.g., 12)> <niters (e.g., 10)> <smooth niters (e.g., 2)> <output_image>","main");
    img  = convertImageToIftImage(image);
    normvalue =  iftNormalizationValue(iftMaximumValue(img)); 
    label     = iftCompute_ISF_MIX_MEAN_Superpixels(img, 3, 0.025, 30, 3, 0, &(nseeds), &(niters));
//     printf("[iftSampling] Ran IFT-ISF for %d iters with %d seeds\n", niters, nseeds);
    border  = iftBorderImage(label);
    int k;
    int v[4];
    int offx, offy, minX, minY, maxX, maxY;
    std::vector<int> xIndexes;
    std::vector<int> yIndexes;
    for(int  j=0; j< image->ny-1; j++){
        for(int  i=0; i< image->nx-1; i++){
//             printf("%02d ", iftImgElem2D(border, i, j));
            k = 4;
            v[0] = iftImgElem2D(border, i, j);
            v[1] = iftImgElem2D(border, i, j+1);
            v[2] = iftImgElem2D(border, i+1, j);
            v[3] = iftImgElem2D(border, i+1, j+1);
            for(int p=0; p<4; p++){
                if(v[p] == 0) k--;
            }
            if(k < 3) continue;
            
            // Trick for ignoring zeroes
            for(int p=0; p<3; p++){
                if(v[p] == 0) v[p] = v[p + 1];
            }
            if(v[3] == 0) v[3] = v[0];
            
            // Expanding all cases to ignore
            if(v[0] == v[1] && v[0] == v[2]) continue;
            if(v[0] == v[1] && v[0] == v[3]) continue;
            if(v[0] == v[2] && v[0] == v[3]) continue;
            if(v[1] == v[2] && v[1] == v[3]) continue;

            if(v[0] == v[1] && v[2] == v[3]) continue;
            if(v[0] == v[2] && v[1] == v[3]) continue;
            if(v[0] == v[3] && v[1] == v[2]) continue;
            
            // Using intersection points as centers for patches
            offx = offy = 0;
            minX = patchSizeX / 2;
            minY = patchSizeY / 2;
            maxX = image->nx - patchSizeX / 2;
            maxY = image->ny - patchSizeY / 2;
            if(i < minX) offx = minX - i;
            if(j < minY) offy = minY - j;
            if(i > maxX) offx = maxX - i;
            if(j > maxY) offy = maxY - j;
            
            xIndexes.push_back(i + offx);
            yIndexes.push_back(j + offy);
        }
//         printf("\n");
    }
//     printf("\n");
//     throw(std::runtime_error("escape"));
    GVector* vector_images;
    if(xIndexes.size() > 0){
        vector_images = createNullVector(xIndexes.size(), sizeof(Image*));
        int x, y;
        for (int i=0; i<xIndexes.size(); i++) {
            x = xIndexes[i] - patchSizeX / 2;
            y = yIndexes[i] - patchSizeY / 2;
            VECTOR_GET_ELEMENT_AS(Image*,vector_images,i) = extractSubImage(image,x,y,patchSizeX,patchSizeY,true);
        }
    }
    else {
        vector_images = randomSampling(image, patchSizeX, patchSizeY, 1);
    }
    iftDestroyImage(&label);
    iftDestroyImage(&img);
    iftDestroyImage(&border);
    return vector_images;
}


