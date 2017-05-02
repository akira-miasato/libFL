//
// Created by deangeli on 3/15/17.
//
#include <stdexcept>
#include "morphology.h"

GrayImage *dilate(GrayImage *image, AdjacencyRelation *AdjRel){
    GrayImage* dilatedImage = createGrayImage(image->ncols,image->nrows);
    int n = image->ncols*image->nrows;

#pragma omp parallel for
    for (int p=0; p < n; p++) {
        //iftVoxel u = iftGetVoxelCoord(img,p);
        dilatedImage->val[p] = image->val[p];
        int pixelX = (p%image->ncols);
        int pixelY = (p/image->ncols);
        for (int i=0; i < AdjRel->n; i++) {
            int adjacentX = pixelX + AdjRel->dx[i];
            int adjacentY = pixelY + AdjRel->dy[i];

            if(isValidPixelCoordinate(image,adjacentX,adjacentY)){
                int adjacentIndex = (adjacentY*image->ncols)+adjacentX;
                if(image->val[adjacentIndex] > dilatedImage->val[p]){
                    dilatedImage->val[p] = image->val[adjacentIndex];
                }
            }
        }
    }


    return(dilatedImage);
}

Image *dilate(Image *image, AdjacencyRelation *AdjRel){
    Image* dilatedImage = createImage(image->nx,image->ny,1);
    dilatedImage->scalingFactor = image->scalingFactor;

#pragma omp parallel for
    for (int p=0; p < image->numberPixels; p++) {
        dilatedImage->channel[0][p] = image->channel[0][p];
        int pixelX = (p%image->nx);
        int pixelY = (p/image->ny);
        for (int i=0; i < AdjRel->n; i++) {
            int adjacentX = pixelX + AdjRel->dx[i];
            int adjacentY = pixelY + AdjRel->dy[i];

            if(isValidPixelCoordinate(image,adjacentX,adjacentY)){
                int adjacentIndex = (adjacentY*image->nx)+adjacentX;
                if(image->channel[0][adjacentIndex] > dilatedImage->channel[0][p]){
                    dilatedImage->channel[0][p] = image->channel[0][adjacentIndex];
                }
            }
        }
    }


    return(dilatedImage);
}


GrayImage *erode(GrayImage *image, AdjacencyRelation *AdjRel){
    GrayImage* erodedImage = createGrayImage(image->ncols,image->nrows);
    int n = image->ncols*image->nrows;

#pragma omp parallel for
    for (int p=0; p < n; p++) {
        //iftVoxel u = iftGetVoxelCoord(img,p);
        erodedImage->val[p] = image->val[p];
        int pixelX = (p%image->ncols);
        int pixelY = (p/image->ncols);
        for (int i=0; i < AdjRel->n; i++) {
            int adjacentX = pixelX + AdjRel->dx[i];
            int adjacentY = pixelY + AdjRel->dy[i];

            if(isValidPixelCoordinate(image,adjacentX,adjacentY)){
                int adjacentIndex = (adjacentY*image->ncols)+adjacentX;
                if(image->val[adjacentIndex] < erodedImage->val[p]){
                    erodedImage->val[p] = image->val[adjacentIndex];
                }
            }
        }
    }
    return(erodedImage);
}

Image *erode(Image *image, AdjacencyRelation *AdjRel){
    Image* erodedImage = createImage(image->nx,image->ny,1);
    erodedImage->scalingFactor = image->scalingFactor;
#pragma omp parallel for
    for (int p=0; p < image->numberPixels; p++) {
        //iftVoxel u = iftGetVoxelCoord(img,p);
        erodedImage->channel[0][p] = image->channel[0][p];
        int pixelX = (p%image->nx);
        int pixelY = (p/image->nx);
        for (int i=0; i < AdjRel->n; i++) {
            int adjacentX = pixelX + AdjRel->dx[i];
            int adjacentY = pixelY + AdjRel->dy[i];

            if(isValidPixelCoordinate(image,adjacentX,adjacentY)){
                int adjacentIndex = (adjacentY*image->nx)+adjacentX;
                if(image->channel[0][adjacentIndex] < erodedImage->channel[0][p]){
                    erodedImage->channel[0][p] = image->channel[0][adjacentIndex];
                }
            }
        }
    }
    return erodedImage;
}




GrayImage *open(GrayImage *image, AdjacencyRelation *AdjRel){
    GrayImage* outputImage = NULL;
    GrayImage* erodedImage = NULL;
    erodedImage = erode(image,AdjRel);
    outputImage = dilate(erodedImage,AdjRel);
    destroyGrayImage(&erodedImage);
    return outputImage;
}

Image *open(Image *image, AdjacencyRelation *AdjRel){
    Image* outputImage = NULL;
    Image* erodedImage = NULL;
    erodedImage = erode(image,AdjRel);
    outputImage = dilate(erodedImage,AdjRel);
    destroyImage(&erodedImage);
    return outputImage;
}

GrayImage *close(GrayImage *image, AdjacencyRelation *AdjRel){
    GrayImage* outputImage = NULL;
    GrayImage* dilatedImage = NULL;
    dilatedImage = dilate(image,AdjRel);
    outputImage = erode(dilatedImage,AdjRel);
    destroyGrayImage(&dilatedImage);
    return outputImage;
}

Image *close(Image *image, AdjacencyRelation *AdjRel){
    Image* outputImage = NULL;
    Image* dilatedImage = NULL;
    dilatedImage = dilate(image,AdjRel);
    outputImage = erode(dilatedImage,AdjRel);
    destroyImage(&dilatedImage);
    return outputImage;
}



GrayImage *topHat(GrayImage *image,AdjacencyRelation *AdjRel){
    GrayImage* outputImage = NULL;
    GrayImage* openedImage = open(image,AdjRel);
    outputImage = imageSubtraction(image,openedImage,false);
    destroyGrayImage(&openedImage);
    return outputImage;
}

Image *topHat(Image *image,AdjacencyRelation *AdjRel){
    Image* outputImage = NULL;
    Image* openedImage = open(image,AdjRel);
    outputImage = imageSubtraction(image,openedImage,false);
    destroyImage(&openedImage);
    return outputImage;
}

GrayImage *bottomHat(GrayImage *image,AdjacencyRelation *AdjRel){
    GrayImage* outputImage = NULL;
    GrayImage* closedImage = close(image,AdjRel);
    outputImage = imageSubtraction(closedImage,image,false);
    destroyGrayImage(&closedImage);
    return outputImage;
}

Image *bottomHat(Image *image,AdjacencyRelation *AdjRel){
    Image* outputImage = NULL;
    Image* closedImage = close(image,AdjRel);
    outputImage = imageSubtraction(closedImage,image,false);
    destroyImage(&closedImage);
    return outputImage;
}

GrayImage *morphologicGradient(GrayImage *image,AdjacencyRelation *AdjRel){
    GrayImage* outputImage = NULL;
    GrayImage* dilatedImage = dilate(image,AdjRel);
    GrayImage* erodedImage = erode(image,AdjRel);
    outputImage = imageSubtraction(dilatedImage,erodedImage,false);
    destroyGrayImage(&dilatedImage);
    destroyGrayImage(&erodedImage);
    return outputImage;
}

Image *morphologicGradient(Image *image,AdjacencyRelation *AdjRel){
    Image* outputImage = NULL;
    Image* dilatedImage = dilate(image,AdjRel);
    Image* erodedImage = erode(image,AdjRel);
    outputImage = imageSubtraction(dilatedImage,erodedImage,false);
    destroyImage(&dilatedImage);
    destroyImage(&erodedImage);
    return outputImage;
}




float *applyGranulometryOnImage(GrayImage *image, float startRadius, float endRadius){
    AdjacencyRelation *adjRel = NULL;
    GrayImage *image1 = NULL;
    GrayImage *image2 = NULL;
    GrayImage *image3 = NULL;
    float *featureVector = NULL;
    float step = 0.1;
    float radius1 = 0;
    float radius2 = startRadius;
    int n1 = 0;
    int n2 = 0;
    int n=0;

    while(radius2 < endRadius){
        n2 = countNumberAdjacentsInCircle(radius2);
        if(n2 > n1){
            n1 = n2;
            n++;
        }
        radius2 += step;
    }

    featureVector = (float*)calloc(n,sizeof(float));
    radius1 = 0;
    radius2 = startRadius;
    n = 0;
    copyGrayImage(image,&image1);
    adjRel = createCircularAdjacency(0);
    copyGrayImage(image,&image1);
    while(radius2 < endRadius){
        n2 = countNumberAdjacentsInCircle(radius2);
        if(n2 > n1){
            resizeCircularAdjacency(&adjRel,radius1);
            printf("%d\n",adjRel->n);
            image1 = open(image,adjRel);
            resizeCircularAdjacency(&adjRel,radius2);
            printf("%d\n",adjRel->n);
            image2 = open(image,adjRel);
            image3 = imageSubtraction(image1,image2,true);
            featureVector[n] = sumUpAllPixelsValues(image3);
            n1 = n2;
            radius1 = radius2;
            destroyGrayImage(&image1);
            destroyGrayImage(&image2);
            destroyGrayImage(&image3);
            n++;
        }
        radius2 += step;
    }

    return featureVector;
}

FeatureVector *applyGranulometryOnImage(GrayImage *image, int k){
    AdjacencyRelation *adjRel = NULL;
    GrayImage *image1 = NULL;
    GrayImage *image2 = NULL;
    GrayImage *image3 = NULL;
    float *featureVector = NULL;
    featureVector = (float*)calloc(k+1,sizeof(float));

    for (int i = 0; i <= k; ++i) {
        resizeLosangeAdjacency(&adjRel,i);
        image1 = open(image,adjRel);
        resizeLosangeAdjacency(&adjRel,i+1);
        image2 = open(image,adjRel);
        image3 = imageSubtraction(image1,image2,true);
        featureVector[i] = sumUpAllPixelsValues(image3);
        destroyGrayImage(&image1);
        destroyGrayImage(&image2);
        destroyGrayImage(&image3);

    }
    FeatureVector* vec = createFeatureVector(featureVector,k+1);
    return vec;
}

FeatureVector *applyGranulometryOnImage(ColorImage *image, int k){
    AdjacencyRelation *adjRel = NULL;
    GrayImage *image1 = NULL;
    GrayImage *image2 = NULL;
    GrayImage *image3 = NULL;
    float *featureVector = NULL;
    featureVector = (float*)calloc(k+1,sizeof(float));
    ColorImage* imageYcbcr = RGBtoYCbCr(image);
    GrayImage* grayImage = extractColorChannelAsGrayImage(imageYcbcr,0);

    for (int i = 0; i <= k; ++i) {
        resizeLosangeAdjacency(&adjRel,i);
        image1 = open(grayImage,adjRel);
        resizeLosangeAdjacency(&adjRel,i+1);
        image2 = open(grayImage,adjRel);
        image3 = imageSubtraction(image1,image2,true);
        featureVector[i] = sumUpAllPixelsValues(image3);
        destroyGrayImage(&image1);
        destroyGrayImage(&image2);
        destroyGrayImage(&image3);

    }
    FeatureVector* vec = createFeatureVector(featureVector,k+1);
    return vec;
}

FeatureVector *applyGranulometryOnImage(Image *image, int k){
    FeatureVector* ret;
    if(image->nchannels == 3){
        ColorImage* img = convertImage2ColorImage(image);
        ret = applyGranulometryOnImage(img, k);
        destroyColorImage(&img);
    }
    else if(image->nchannels == 1){
        GrayImage* img = convertImage2GrayImage(image);
        ret = applyGranulometryOnImage(img, k);
        destroyGrayImage(&img);
    }
    else{
        throw std::runtime_error("Image is not grayscale nor RGB!");
    }
    return ret;
}


FeatureVector *getMorphologicalPdf(GrayImage *image, int k){
    AdjacencyRelation *adjRel = NULL;
    GrayImage *image0 = NULL;
    GrayImage *imageJ = NULL;
    float *featureVector = NULL;
    featureVector = (float*)calloc(k,sizeof(float));
    resizeLosangeAdjacency(&adjRel,0);
    image0= open(image,adjRel);
    float v0 = sumUpAllPixelsValues(image0);
    destroyGrayImage(&image0);

    for (int i = 1; i <= k; ++i) {
        resizeLosangeAdjacency(&adjRel,i);
        imageJ = open(image,adjRel);
        float vJ = sumUpAllPixelsValues(imageJ);
        float feature = 1 - (vJ/v0);
        featureVector[i-1] = feature;

        destroyGrayImage(&imageJ);
    }
    FeatureVector *vector = createFeatureVector(featureVector,k);
    return vector;
}

FeatureVector *getMorphologicalPdf(ColorImage *image, int k){
    AdjacencyRelation *adjRel = NULL;
    GrayImage *image0 = NULL;
    GrayImage *imageJ = NULL;
    float *featureVector = NULL;
    featureVector = (float*)calloc(k,sizeof(float));
    resizeLosangeAdjacency(&adjRel,0);
    ColorImage* imageYcbcr = RGBtoYCbCr(image);
    GrayImage* grayImage = extractColorChannelAsGrayImage(imageYcbcr,0);
    destroyColorImage(&imageYcbcr);
    image0= open(grayImage,adjRel);
    float v0 = sumUpAllPixelsValues(image0);
    destroyGrayImage(&image0);

    for (int i = 1; i <= k; ++i) {
        resizeLosangeAdjacency(&adjRel,i);
        imageJ = open(grayImage,adjRel);
        float vJ = sumUpAllPixelsValues(imageJ);
        float feature = 1 - (vJ/v0);
        featureVector[i-1] = feature;
        destroyGrayImage(&imageJ);
    }
    FeatureVector *vector = createFeatureVector(featureVector,k);
    return vector;
}

FeatureVector *getMorphologicalPdf(Image *image, int k){
    FeatureVector* ret;
    if(image->nchannels == 3){
        ColorImage* img = convertImage2ColorImage(image);
        ret = getMorphologicalPdf(img, k);
        destroyColorImage(&img);
    }
    else if(image->nchannels == 1){
        GrayImage* img = convertImage2GrayImage(image);
        ret = getMorphologicalPdf(img, k);
        destroyGrayImage(&img);
    }
    else{
        throw std::runtime_error("Image is not grayscale nor RGB!");
    }
    return ret;
}


Image* transformAdjacencyRelation2Image(AdjacencyRelation *adjRel,int nx,int ny,int centerX,int centerY){
    Image* image = createImage(nx,ny,1);
    image->scalingFactor = 255;
    int coordinateX = 0;
    int coordinateY = 0;
    int index;
    for (int i = 0; i < adjRel->n; ++i) {
        coordinateX = adjRel->dx[i] + centerX;
        coordinateY = adjRel->dy[i] + centerY;
        index = (coordinateY*nx) +  coordinateX;
        if(isValidPixelCoordinate(image,coordinateX,coordinateY)){
            image->channel[0][index] = image->scalingFactor;
        }
    }
    return image;
}

AdjacencyRelation* transformImage2AdjacencyRelation(Image *image, float thresholding,int centerX,int centerY){

    int counter = 0;
    for (int p = 0; p < image->numberPixels; ++p) {
        if(image->channel[0][p] > thresholding){
            counter++;
        }
    }
    AdjacencyRelation *adjacencyRelation = createAdjacencyRelation(counter);
    int k = 0;

    for (int p = 0; p < image->numberPixels; ++p) {

        if(image->channel[0][p] > thresholding) {
            int dx = (p % image->nx);
            int dy = (p / image->nx);
            dx -= centerX;
            dy -= centerY;


            if (abs(dx) > adjacencyRelation->maxDx) {
                adjacencyRelation->maxDx = abs(dx);
            }
            if (abs(dy) > adjacencyRelation->maxDy) {
                adjacencyRelation->maxDy = abs(dy);
            }
            int k_copy;

            k_copy = k;
            k++;

            adjacencyRelation->dx[k_copy] = dx;
            adjacencyRelation->dy[k_copy] = dy;
        }

    }
    return adjacencyRelation;
}

AdjacencyRelation* dilate(AdjacencyRelation *adjacencyRelation1, AdjacencyRelation *adjacencyRelation2){
    int nx = ((adjacencyRelation1->maxDx*2 + 1) + (adjacencyRelation2->maxDx*2 + 1)) - 1;
    int ny = ((adjacencyRelation1->maxDy*2 + 1) + (adjacencyRelation2->maxDy*2 + 1)) -1;
    int centerX = nx/2;
    int centerY = ny/2;
    Image* imageStructElement = transformAdjacencyRelation2Image(adjacencyRelation2,nx,ny,centerX,centerY);
    Image* imageDilated = dilate(imageStructElement,adjacencyRelation1);
    AdjacencyRelation* adjacencyRelationDilated = transformImage2AdjacencyRelation(imageDilated, imageDilated->scalingFactor/2,centerX,centerY);
    destroyImage(&imageStructElement);
    destroyImage(&imageDilated);
    return adjacencyRelationDilated;
}

FeatureVector *getMorphologicalPdf(Image *image, AdjacencyRelation* adjacencyRelation, int k_times){

    FeatureVector *featureVector = NULL;
    if(k_times < 0){
        return featureVector;
    }
    featureVector = createFeatureVector(k_times+1);
    featureVector->features[0] = 0;
    float v0 = sumUpAllPixelsValues(image,true);
    v0 += 0.00000001;//somo isso para evitar divisao por zero
    float vk = 0;
    float feature = 0;
    Image* openedImage = NULL;
    AdjacencyRelation* currentStructorElement = copyAdjcencyRelation(adjacencyRelation);

    for (int i = 1; i <= k_times; ++i) {
        openedImage = open(image,currentStructorElement);
        vk = sumUpAllPixelsValues(openedImage,true);
        feature = 1 - (vk/v0);
        featureVector->features[i] = feature;
        destroyImage(&openedImage);
        AdjacencyRelation* auxStructorElement = dilate(adjacencyRelation,currentStructorElement);
        copyAdjcencyRelationInPlace(auxStructorElement,&currentStructorElement);
        destroyAdjacencyRelation(&auxStructorElement);
    }

    //FeatureVector *vector = createFeatureVector(featureVector,k);
    return featureVector;
}

