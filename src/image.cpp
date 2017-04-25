/*
 *Created by Deangeli Gomes Neves
 *
 * This software may be freely redistributed under the terms
 * of the MIT license.
 *
 */

#include "image.h"


GrayImage *createGrayImage(int ncols, int nrows)
{
    GrayImage *img=NULL;
    int i;

    img = (GrayImage *) calloc(1,sizeof(GrayImage));
    if (img == NULL){
        Error(MSG1,"CreateImage");
    }

    img->val   = AllocIntArray(nrows*ncols);
    img->tbrow = AllocIntArray(nrows);

    img->tbrow[0]=0;
    for (i=1; i < nrows; i++)
        img->tbrow[i]=img->tbrow[i-1]+ncols;
    img->ncols = ncols;
    img->nrows = nrows;

    return(img);
}

void destroyGrayImage(GrayImage **img)
{
    GrayImage *aux;

    aux = *img;
    if(aux != NULL){
        if (aux->val != NULL)   free(aux->val);
        if (aux->tbrow != NULL) free(aux->tbrow);
        free(aux);
        *img = NULL;
    }
}

GrayImage *readGrayImage(char *filename)
{
    FILE *fp=NULL;
    unsigned char *value=NULL;
    char type[10];
    int  i,ncols,nrows,n;
    GrayImage *img = NULL;
    char z[256];

    fp = fopen(filename,"rb");
    if (fp == NULL){
        fprintf(stderr,"Cannot open %s\n",filename);
        exit(-1);
    }
    fscanf(fp,"%s\n",type);
    if((strcmp(type,"P5")==0)){
        NCFgets(z,255,fp);
        sscanf(z,"%d %d\n",&ncols,&nrows);
        n = ncols*nrows;
        NCFgets(z,255,fp);
        sscanf(z,"%d\n",&i);
        fgetc(fp);
        value = (unsigned char *)calloc(n,sizeof(unsigned char));
        if (value != NULL){
            fread(value,sizeof(unsigned char),n,fp);
        }else{
            fprintf(stderr,"Insufficient memory in ReadImage\n");
            exit(-1);
        }
        fclose(fp);
        img = createGrayImage(ncols,nrows);
        for (i=0; i < n; i++)
            img->val[i]=(int)value[i];
        free(value);
    }else{
        if((strcmp(type,"P2")==0)){
            NCFgets(z,255,fp);
            sscanf(z,"%d %d\n",&ncols,&nrows);
            n = ncols*nrows;
            NCFgets(z,255,fp);
            sscanf(z,"%d\n",&i);
            img = createGrayImage(ncols,nrows);
            for (i=0; i < n; i++)
                fscanf(fp,"%d",&img->val[i]);
            fclose(fp);
        }else{
            fprintf(stderr,"Input image must be P2 or P5\n");
            exit(-1);
        }
    }

    return(img);
}

void writeGrayImage(GrayImage *img,char *filename)
{
    FILE *fp;
    int i, n, Imax,Imin;

    fp = fopen(filename,"wb");
    if (fp == NULL){
        fprintf(stderr,"Cannot open %s\n",filename);
        exit(-1);
    }
    n    = img->ncols*img->nrows;
    Imax = INT_MIN; Imin = INT_MAX;
    for (i=0; i < n; i++) {

        if (img->val[i] > Imax)
            Imax = img->val[i];
        if (img->val[i] < Imin)
            Imin = img->val[i];
    }


    fprintf(fp,"P2\n");
    fprintf(fp,"%d %d\n",img->ncols,img->nrows);
    fprintf(fp,"%d\n",Imax-Imin);

    for (i=0; i < n; i++) {
        fprintf(fp,"%d ",img->val[i]-Imin);
        if (((i+1)%17) == 0)
            fprintf(fp,"\n");
    }

    fclose(fp);
}


int minimumValue(GrayImage *img){
    int i,min;

    min = img->val[0];
    int n = img->nrows*img->ncols;
    for (i = 0; i < n; i++){
        if (img->val[i] < min) {
            min = img->val[i];
        }
    }
    return min;
}
int maximumValue(GrayImage *img){
    int i,max;

    max = img->val[0];
    int n = img->nrows*img->ncols;
    for (i = 0; i < n; i++){
        if (img->val[i] > max) {
            max = img->val[i];
        }
    }
    return max;
}

ColorImage *createColorImage(int nx, int ny)
{
    ColorImage *I=(ColorImage *)calloc(1,sizeof(ColorImage));
    int y;

    I->nx = nx;
    I->ny = ny;
    I->dx = 1.0;
    I->dy = 1.0;
    sprintf(I->unid,"mm");

    I->cor = (Cor **)calloc(ny,sizeof(Cor *));
    if (I->cor == NULL)
        Error(MSG1,"createColorImage");
    else{
        for (y=0; y < I->ny; y++){
            I->cor[y] = (Cor *)calloc(nx,sizeof(Cor));
            if (I->cor[y] == NULL)
                Error(MSG1,"createColorImage");
        }
    }
    return(I);
}

ColorImage *readColorImage(char *filename)
{
    FILE *fp=NULL;
    char type[10];
    int  x,y,nx,ny,Imax;
    ColorImage *I=NULL;
    char remarks[256];
    ushort rgb16[3];

    fp = fopen(filename,"r");
    if (fp != NULL) {
        fscanf(fp,"%s",type);
        NCFgets(remarks,255,fp);

        if (strcmp(type,"P6")==0){
            int c = getc(fp);
            while (c == '#') {
                while (getc(fp) != '\n');
                c = getc(fp);
            }
            ungetc(c, fp);
            if (fscanf(fp, "%d %d", &nx, &ny) != 2) {
                fprintf(stderr, "Invalid image size (error loading '%s')\n", filename);
                exit(1);
            }

            I = createColorImage(nx,ny);

            if (fscanf(fp, "%d", &Imax) != 1) {
                fprintf(stderr, "Invalid rgb component (error loading '%s')\n",
                        filename);
                exit(1);
            }

            I->Imax = Imax;
            while (fgetc(fp) != '\n');

            if (Imax <= 255){
                for (y=0; y < ny; y++)
                    for (x=0; x < nx; x++){
                        I->cor[y][x].val[RED]=(int)fgetc(fp);//red
                        I->cor[y][x].val[GREEN]=(int)fgetc(fp);//green
                        I->cor[y][x].val[BLUE]=(int)fgetc(fp);//blue
                    }
            }else if (Imax > 255 && Imax <= 65535){
                int rgbBitDepth = 9;

                while ((1 << rgbBitDepth) <= Imax) {
                    rgbBitDepth++;
                }
                for (y=0; y < ny; y++) {
                    for (x=0; x < nx; x++){
                        if (fread(rgb16, 2, 3, fp) == 3) {
                            I->cor[y][x].val[RED]=((rgb16[0] & 0xff) << 8) | ((ushort) rgb16[0] >> 8);
                            I->cor[y][x].val[GREEN]=((rgb16[1] & 0xff) << 8) | ((ushort) rgb16[1] >> 8);
                            I->cor[y][x].val[BLUE]=((rgb16[2] & 0xff) << 8) | ((ushort) rgb16[2] >> 8);
                        } else{
                            Error("Reading 16-bit error","ReadColorImage");
                        }
                    }
                }
            }
        }else if(strcmp(type,"P3")==0){
            int c = getc(fp);
            while (c == '#') {
                while (getc(fp) != '\n');
                c = getc(fp);
            }
            ungetc(c, fp);
            if (fscanf(fp, "%d %d", &nx, &ny) != 2) {
                fprintf(stderr, "Invalid image size (error loading '%s')\n", filename);
                exit(1);
            }

            I = createColorImage(nx,ny);

            if (fscanf(fp, "%d", &Imax) != 1) {
                fprintf(stderr, "Invalid rgb component (error loading '%s')\n",
                        filename);
                exit(1);
            }

            I->Imax = Imax;

            int R=0,G=0,B=0;

            for (y=0; y < ny; y++) {
                for (x=0; x < nx; x++){
                    (void)fscanf(fp, "%d %d %d",&R,&G,&B);
                    I->cor[y][x].val[0] = R;
                    I->cor[y][x].val[1] = G;
                    I->cor[y][x].val[2] = B;
                }
            }
        }

        else{
            Error("ColorImage type invalid","ReadColormage");
        }

        fclose(fp);
    }else{
        Error(MSG2,"ReadColorImage");
    }

    return(I);
}

void writeColorImage(ColorImage *I, char *filename)
{
    FILE *fp=NULL;
    int  x,y;
    ushort rgb16[3];

    fp = fopen(filename,"w");
    if (fp != NULL) {
        fprintf(fp,"P6\n");
        fprintf(fp,"%d %d\n",I->nx,I->ny);

        int max_val = maximumColorValue(I);
        int min_val = minimumColorValue(I);

        if (min_val < 0){
            Error(MSG2,"WriteColorImage");
            return;
        }

        if (max_val < 256){
            fprintf(fp,"255\n");
            for (y=0; y < I->ny; y++)
                for (x=0; x < I->nx; x++){
                    fputc((uchar)I->cor[y][x].val[RED],fp);//red
                    fputc((uchar)I->cor[y][x].val[GREEN],fp);//green
                    fputc((uchar)I->cor[y][x].val[BLUE],fp);//blue
                }
        } else if (max_val < 65536){
            int rgbBitDepth = 9;
            // find the bit depth for the maximum value img_max_val
            while ((1 << rgbBitDepth) <= max_val) {
                rgbBitDepth++;
            }

            fprintf(fp, "%d\n", (1 << rgbBitDepth) - 1);
            for (y=0; y < I->ny; y++)
                for (x=0; x < I->nx; x++){
                    rgb16[RED] = ((I->cor[y][x].val[RED] & 0xff) << 8) | ((ushort) I->cor[y][x].val[RED] >> 8);//I->cor[y][x].val[RED];
                    rgb16[GREEN] = ((I->cor[y][x].val[GREEN] & 0xff) << 8) | ((ushort) I->cor[y][x].val[GREEN] >> 8);//I->cor[y][x].val[GREEN];
                    rgb16[BLUE] = ((I->cor[y][x].val[BLUE] & 0xff) << 8) | ((ushort) I->cor[y][x].val[BLUE] >> 8);//I->cor[y][x].val[BLUE];
                    // write 6 bytes for each image pixel
                    if (fwrite(rgb16, 2, 3, fp) != 3) {
                        Error("Cannot write 16-bit image as P6", "iftWriteImageP6");
                    }
                }
        }
        fclose(fp);
    }else{
        Error(MSG2,"WriteColorImage");
    }

}

void destroyColorImage(ColorImage **I)
{
    int y;

    if ((*I) != NULL) {
        for (y=0; y < (*I)->ny; y++)
            free((*I)->cor[y]);
        free((*I)->cor);
        free(*I);
        *I = NULL;
    }
}


int minimumColorValue(ColorImage *img){

    int img_min_val[3];
    for (int i=0; i<3; i++){
        img_min_val[i] = minimumIntensityColor(img,i);
    }

    return MIN(MIN(img_min_val[RED],img_min_val[GREEN]),img_min_val[BLUE]);
}

int maximumColorValue(ColorImage *img){

    int img_max_val[3];
    for (int i=0; i<3; i++){
        img_max_val[i] = maximumIntensityColor(img,i);
    }

    return MAX(MAX(img_max_val[RED],img_max_val[GREEN]),img_max_val[BLUE]);
}



int minimumIntensityColor(ColorImage *img, int c){
    int i,j,min;

    min = INT_MAX;
    for (i = 0; i<img->nx; i++)
        for (j=0; j<img->ny; j++)
            if (img->cor[j][i].val[c] < min)
                min = img->cor[j][i].val[c];

    return min;
}

int maximumIntensityColor(ColorImage *img, int c){
    int i,j,max;

    max = -1;
    for (i = 0; i<img->nx; i++)
        for (j=0; j<img->ny; j++)
            if (img->cor[j][i].val[c] > max)
                max = img->cor[j][i].val[c];

    return max;
}

bool isValidPixelCoordinate(GrayImage *image,int pixelCoordinateX,int pixelCoordinateY){
    if(pixelCoordinateX < 0 || pixelCoordinateY < 0){
        return false;
    }
    if(pixelCoordinateX >= image->ncols || pixelCoordinateY >= image->ncols){
        return false;
    }
    return true;
}

bool isValidPixelCoordinate(Image *image,int pixelCoordinateX,int pixelCoordinateY){
    if(pixelCoordinateX < 0 || pixelCoordinateY < 0){
        return false;
    }
    if(pixelCoordinateX >= image->nx || pixelCoordinateY >= image->ny){
        return false;
    }
    return true;
}

bool isImagesSameDomain(GrayImage *image1,GrayImage *image2){

    if(image1->nrows == image2->nrows && image1->ncols == image2->ncols){
        return true;
    }
    return false;
}

bool isImagesSameDomain(Image *image1,Image *image2){

    if(image1->nx == image2->nx && image1->nx == image2->nx){
        if(image1->nchannels == image2->nchannels){
            return true;
        }
    }
    return false;
}

void copyGrayImage(GrayImage *image1,GrayImage **image2){
    if((*image2) == NULL){
        *image2 = createGrayImage(image1->ncols,image1->nrows);
    }

    if(isImagesSameDomain(image1,*image2)){
        for (int n = 0; n < image1->nrows*image1->ncols; ++n) {
            (*image2)->val[n] = image1->val[n];
        }
    }
    else{
        printf("different domains");
    }
}

GrayImage *imageSubtraction(GrayImage *image1, GrayImage *image2, bool saturation){
    GrayImage *outputImage = NULL;
    if(isImagesSameDomain(image1,image2)){
        outputImage = createGrayImage(image1->ncols,image1->nrows);
        int n = image1->ncols*image1->nrows;
        int result;
        for (int i = 0; i < n; ++i) {
            result = image1->val[i] - image2->val[i];
            if(saturation){
                result = (result<0)?0:result;
            }
            outputImage->val[i] = result;
        }
    }
    return outputImage;
}



int sumUpAllPixelsValues(GrayImage *image){
    int sum = 0;
    for (int n = 0; n < image->nrows*image->ncols; ++n) {
        sum += image->val[n];
    }
    return sum;
}


void writeImageAsASCII(GrayImage* image,const char *filename,unsigned int maxValueOnChanel){
    int i, j;
    FILE *fp = fopen(filename, "w");
    (void) fprintf(fp, "P2\n%d %d\n%d\n", image->ncols,image->nrows, (int) maxValueOnChanel);
    int k = 0;
    for (i = 0; i < image->nrows; i++){
        for (j = 0; j < image->ncols; j++){
            (void) fprintf(fp,"%d ",image->val[k]);
            k++;
        }
        (void) fprintf(fp,"\n");
    }
    (void) fclose(fp);
}

void writeImageAsASCII(ColorImage* image,const char *filename, unsigned int maxValueOnChanel){
    int i, j;
    FILE *fp = fopen(filename, "w");
    (void) fprintf(fp, "P3\n%d %d\n%d\n", image->nx,image->ny, (int) maxValueOnChanel);
    for (i = 0; i < image->ny; i++){
        for (j = 0; j < image->nx; j++){
            (void) fprintf(fp,"%d %d %d ",image->cor[i][j].val[0],image->cor[i][j].val[1],image->cor[i][j].val[2]);
        }
        (void) fprintf(fp,"\n");
    }
    (void) fclose(fp);
}

void writeImageAsBinary(GrayImage* image,const char *filename,unsigned int maxValueOnChanel){
    int i, j;
    FILE *fp = fopen(filename, "wb");
    (void) fprintf(fp, "P5\n%d %d\n%d\n", image->ncols,image->nrows, (int) maxValueOnChanel);
    int k = 0;
    for (i = 0; i < image->nrows; i++){
        for (j = 0; j < image->ncols; j++){
            static unsigned char gray[1];
            gray[0] = image->val[k];
            (void) fwrite(gray, 1, 1, fp);
            k++;
        }
        //(void) fprintf(fp,"\n");
    }
    (void) fclose(fp);
}

void writeImageAsBinary(ColorImage* image,const char *filename, unsigned int maxValueOnChanel){
    int i, j;
    FILE *fp = fopen(filename, "wb");
    (void) fprintf(fp, "P6\n%d %d\n%d\n", image->nx,image->ny, (int) maxValueOnChanel);
    for (i = 0; i < image->ny; i++){
        for (j = 0; j < image->nx; j++){
            static unsigned char color[3];
            color[0] = image->cor[i][j].val[0];
            color[1] = image->cor[i][j].val[1];
            color[2] = image->cor[i][j].val[2];
            (void) fwrite(color, 1, 3, fp);
        }
        //(void) fprintf(fp,"\n");
    }
    (void) fclose(fp);
}

void writeImage(ColorImage* image,const char *filename, const char *magicNumber,unsigned int maxValueOnChanel){
    //ASCII
    if (strcmp(magicNumber,"P3")==0){
        writeImageAsASCII(image,filename,maxValueOnChanel);
    }//Binary
    else if(strcmp(magicNumber,"P6")==0){
        writeImageAsBinary(image,filename,maxValueOnChanel);
    }else{
        fprintf(stderr,"magic number desconhecido. Por favor, acesse https://en.wikipedia.org/wiki/Netpbm_format "
                "para saber mais sobre");
    }
}

void writeImage(GrayImage* image,const char *filename, const char *magicNumber,unsigned int maxValueOnChanel){
    //ASCII
    if (strcmp(magicNumber,"P2")==0){
        writeImageAsASCII(image,filename,maxValueOnChanel);
    }//Binary
    else if(strcmp(magicNumber,"P5")==0){
        writeImageAsBinary(image,filename,maxValueOnChanel);
    }else{
        fprintf(stderr,"magic number desconhecido. Por favor, acesse https://en.wikipedia.org/wiki/Netpbm_format "
                "para saber mais sobre");
    }
}

void writeImage(GrayImage* image,const char *filename){
    writeImage(image,filename, "P5",255);
}

void writeImage(ColorImage* image,const char *filename){
    writeImage(image,filename, "P6",255);
}

ColorImage *RGBtoYCbCr(ColorImage *rgb)
{
    ColorImage *ycbcr;
    float a = 16.0;
    float b = 128.0;

    ycbcr = createColorImage(rgb->nx,rgb->ny);

    for (int y=0; y < rgb->ny; y++)
        for (int x=0; x < rgb->nx; x++){
            float lum  = (0.257*(float)rgb->cor[y][x].val[0]+
                          0.504*(float)rgb->cor[y][x].val[1]+
                          0.098*(float)rgb->cor[y][x].val[2]+a);
            float Cb = (-0.148*(float)rgb->cor[y][x].val[0]+
                        -0.291*(float)rgb->cor[y][x].val[1]+
                        0.439*(float)rgb->cor[y][x].val[2]+b);
            float Cr = (0.439*(float)rgb->cor[y][x].val[0]+
                        -0.368*(float)rgb->cor[y][x].val[1]+
                        -0.071*(float)rgb->cor[y][x].val[2]+b);

            if (lum < 0)   lum = 0.0;
            if (lum > 255) lum = 255;
            if (Cb < 0)   Cb = 0.0;
            if (Cb > 255) Cb = 255;
            if (Cr < 0)   Cr = 0.0;
            if (Cr > 255) Cr = 255;

            ycbcr->cor[y][x].val[0] = lum;
            ycbcr->cor[y][x].val[1] = Cb;
            ycbcr->cor[y][x].val[2] = Cr;

        }

    return(ycbcr);
}

GrayImage* extractColorChannelAsGrayImage(ColorImage* colorImage,int channel){
    GrayImage* grayImage= createGrayImage(colorImage->nx,colorImage->ny);
    int k=0;
    for (int i = 0; i < colorImage->ny; ++i) {
        for (int j = 0; j < colorImage->nx; ++j) {
            grayImage->val[k] = colorImage->cor[i][j].val[channel];
            k++;
        }
    }
    return grayImage;
}

Image* createImage(int nx, int ny,int nchannels){
    Image* image = (Image*)calloc(1,sizeof(Image));
    image->nx = nx;
    image->ny = ny;
    image->nz = 1;
    image->numberPixels = image->nx*image->ny*image->nz;
    image->dx = 0;
    image->dy = 0;
    image->dz = 0;
    image->scalingFactor = 255;
    image->channelDepth = 8;
    image->unid[0] = 'm';
    image->unid[1] = 'm';
    image->channel = (float**)calloc(nchannels,sizeof(float*));
    image->nchannels = nchannels;
    image->colorSpace = UNKNOWN;
    image->dataTypeId = FLOAT;
    for (int i = 0; i < image->nchannels; ++i) {
        image->channel[i] = (float*)calloc(image->numberPixels,sizeof(float));
        //image->channel[i] = (float*)calloc(image->numberPixels,sizeof(float*));
        //image->channel->floatChannel[i] = (float*)calloc(image->numberPixels,sizeof(float));
    }
    return image;
}

Image* createImage(int nx, int ny,int nz, int nchannels){
    Image* image = (Image*)calloc(1,sizeof(Image));
    image->nx = nx;
    image->ny = ny;
    image->nz = nz;
    image->numberPixels = image->nx*image->ny*image->nz;
    image->dx = 0;
    image->dy = 0;
    image->dz = 0;
    image->scalingFactor = 255;
    image->channelDepth = 8;
    image->unid[0] = 'm';
    image->unid[1] = 'm';
    image->channel = (float**)calloc(nchannels,sizeof(float*));
    image->nchannels = nchannels;
    image->colorSpace = UNKNOWN;
    image->dataTypeId = FLOAT;
    for (int i = 0; i < image->nchannels; ++i) {
        image->channel[i] = (float*)calloc(image->numberPixels,sizeof(float));
        //image->channel[i] = (float*)calloc(image->numberPixels,sizeof(float*));
        //image->channel->floatChannel[i] = (float*)calloc(image->numberPixels,sizeof(float));
    }
    return image;
}



Image* createImage(int nx, int ny){
    Image* image = (Image*)calloc(1,sizeof(Image));
    image->nx = nx;
    image->ny = ny;
    image->numberPixels = nx*ny;
    image->dx = 0;
    image->dy = 0;
    image->scalingFactor = 255;
    image->channelDepth = 8;
    image->unid[0] = 'm';
    image->unid[1] = 'm';
    image->channel = (float**)calloc(1,sizeof(float*));
    image->nchannels = 1;
    image->colorSpace = UNKNOWN;
    image->dataTypeId = FLOAT;
    for (int i = 0; i < image->nchannels; ++i) {
        image->channel[i] = (float*)calloc(image->numberPixels,sizeof(float*));
    }
    return image;
}

void destroyImage(Image**image ){
    Image* auxPointer = *image;
    if(auxPointer == NULL){
        return;
    }
    for (int i = 0; i < auxPointer->nchannels; ++i) {
        free(auxPointer->channel[i]);
    }
    free(auxPointer);
    auxPointer = NULL;
}

Image* readImage(char *filename){
    char ext[10],*pos;
    pos = strrchr(filename,'.') + 1;
    sscanf(pos,"%s",ext);
    Image* image = NULL;
    //gray image
    if (strcmp(ext,"pgm")==0){
        image = readImagePGM(filename);
    }else if(strcmp(ext,"ppm")==0){
        image = readImagePPM(filename);
    }else if(strcmp(ext,"png")==0){
        image = readImagePNG(filename);
    }
    else{
        printf("image format unknown\n");
    }
    return image;
}

Image *readImagePGM(char *filename)
{
    FILE *fp=NULL;
    unsigned char *value=NULL;
    char type[10];
    int  maxValue,ncols,nrows,n;
    Image *img = NULL;
    char z[256];

    fp = fopen(filename,"rb");
    if (fp == NULL){
        fprintf(stderr,"Cannot open %s\n",filename);
        exit(-1);
    }
    fscanf(fp,"%s\n",type);
    NCFgets(z,255,fp);
    if((strcmp(type,"P5")==0)){
        int c = getc(fp);
        while (c == '#') {
            while (getc(fp) != '\n');
            c = getc(fp);
        }
        ungetc(c, fp);
        sscanf(z,"%d %d\n",&ncols,&nrows);
        n = ncols*nrows;
        NCFgets(z,255,fp);
        sscanf(z,"%d\n",&maxValue);
        fgetc(fp);
        value = (unsigned char *)calloc(n,sizeof(unsigned char));
        if (value != NULL){
            fread(value,sizeof(unsigned char),n,fp);
        }else{
            fprintf(stderr,"Insufficient memory in ReadImage\n");
            exit(-1);
        }
        fclose(fp);
        img = createImage(ncols,nrows,1);
        img->scalingFactor = maxValue;
        double log2 = 0.30102999566;
        img->channelDepth = (log(maxValue+1)/log2)+0.4;
        img->colorSpace = GRAYSCALE;
        for (int i=0; i < n; i++)
            img->channel[0][i]=(float)value[i];
        free(value);
    }else{
        if((strcmp(type,"P2")==0)){
            int c = getc(fp);
            while (c == '#') {
                while (getc(fp) != '\n');
                c = getc(fp);
            }
            ungetc(c, fp);
            sscanf(z,"%d %d\n",&ncols,&nrows);
            n = ncols*nrows;
            NCFgets(z,255,fp);
            sscanf(z,"%d\n",&maxValue);
            img = createImage(ncols,nrows,1);
            img->scalingFactor = maxValue;
            double log2 = 0.30102999566;
            img->channelDepth = (log(maxValue+1)/log2)+0.4;
            img->colorSpace = GRAYSCALE;

            for (int i=0; i < n; i++)
                fscanf(fp,"%f",&img->channel[0][i]);
            fclose(fp);
        }else{
            fprintf(stderr,"Input image must be P2 or P5\n");
            exit(-1);
        }
    }
    return(img);
}

Image *readImagePPM(char *filename){
    FILE *fp=NULL;
    char type[10];
    int  x,y,nx,ny,Imax;
    Image *I=NULL;
    ushort rgb16[3];
    char z[256];

    fp = fopen(filename,"r");
    if (fp != NULL) {
        fscanf(fp,"%s",type);
        NCFgets(z,255,fp);

        if (strcmp(type,"P6")==0){
            int c = getc(fp);
            while (c == '#') {
                while (getc(fp) != '\n');
                c = getc(fp);
            }
            ungetc(c, fp);

            if (fscanf(fp, "%d %d", &nx, &ny) != 2) {
                fprintf(stderr, "Invalid image size (error loading '%s')\n", filename);
                exit(1);
            }
            I = createImage(nx,ny,3);

            if (fscanf(fp, "%d", &Imax) != 1) {
                fprintf(stderr, "Invalid rgb component (error loading '%s')\n",
                        filename);
                exit(1);
            }

            I->scalingFactor = Imax;
            double log2 = 0.30102999566;
            I->channelDepth = (log(Imax+1)/log2)+0.4;
            I->colorSpace = RGB;
            while (fgetc(fp) != '\n');

            if (Imax <= 255){
                int iter = 0;
                for (y=0; y < ny; y++) {
                    for (x = 0; x < nx; x++) {
                        I->channel[0][iter] =  (float)fgetc(fp);//red
                        I->channel[1][iter] =  (float)fgetc(fp);//green
                        I->channel[2][iter] =  (float)fgetc(fp);//blue
                        iter++;
                    }
                }
            }else if (Imax > 255 && Imax <= 65535){

                int iter = 0;
                for (y=0; y < ny; y++) {
                    for (x=0; x < nx; x++){
                        if (fread(rgb16, 2, 3, fp) == 3) {
                            I->channel[0][iter]=(((rgb16[0] & 0xff) << 8) | ((ushort) rgb16[0] >> 8));
                            I->channel[1][iter]=(((rgb16[1] & 0xff) << 8) | ((ushort) rgb16[1] >> 8));
                            I->channel[2][iter]=(((rgb16[2] & 0xff) << 8) | ((ushort) rgb16[2] >> 8));
                        } else{
                            Error("Reading 16-bit error","ReadColorImage");
                        }
                        iter++;
                    }
                }
            }
        }else if(strcmp(type,"P3")==0){
            int c = getc(fp);
            while (c == '#') {
                while (getc(fp) != '\n');
                c = getc(fp);
            }
            ungetc(c, fp);
            if (fscanf(fp, "%d %d", &nx, &ny) != 2) {
                fprintf(stderr, "Invalid image size (error loading '%s')\n", filename);
                exit(1);
            }

            I = createImage(nx,ny,3);
            I->colorSpace = RGB;

            if (fscanf(fp, "%d", &Imax) != 1) {
                fprintf(stderr, "Invalid rgb component (error loading '%s')\n",
                        filename);
                exit(1);
            }

            I->scalingFactor = Imax;
            double log2 = 0.30102999566;
            I->channelDepth = (log(Imax+1)/log2)+0.4;
            int R=0,G=0,B=0;
            int iter=0;
            for (y=0; y < ny; y++) {
                for (x=0; x < nx; x++){
                    (void)fscanf(fp, "%d %d %d",&R,&G,&B);
                    I->channel[0][iter] = R;
                    I->channel[1][iter] = G;
                    I->channel[2][iter] = B;
                    iter++;
                }
            }
        }

        else{
            Error("ColorImage type invalid","ReadColormage");
        }

        fclose(fp);
    }else{
        Error(MSG2,"ReadColorImage");
    }

    return(I);

}


Image *readImagePNG(char *filename){
/*
 * Copyright 2002-2010 Guillaume Cottenceau.
 *
 * This software may be freely redistributed under the terms
 * of the X11 license.
 *
 */
    Image* image = NULL;
    int width, height;
    //png_byte color_type;
    png_byte bit_depth;

    png_structp png_ptr;
    png_infop info_ptr;
    //int number_of_passes;
    png_bytep * row_pointers;
    char header[8];    // 8 is the maximum size that can be checked

    /* open file and test for it being a png */
    FILE *fp = fopen(filename, "rb");
    if (!fp){
        printf("[read_png_file] File %s could not be opened for reading\n", filename);
        return image;
    }
    fread(header, 1, 8, fp);
    if (png_sig_cmp(((const png_byte*)header), 0, 8)){
        printf("[read_png_file] File %s is not recognized as a PNG file\n",filename);
        return image;
    }


    /* initialize stuff */
    png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);

    if (!png_ptr){
        printf("[read_png_file] png_create_read_struct failed\n");
        return image;
    }

    info_ptr = png_create_info_struct(png_ptr);
    if (!info_ptr){
        printf("[read_png_file] png_create_info_struct failed\n");
        return image;
    }

    if (setjmp(png_jmpbuf(png_ptr))) {
        printf("[read_png_file] Error during init_io\n");
        return image;
    }

    png_init_io(png_ptr, fp);
    png_set_sig_bytes(png_ptr, 8);

    png_read_info(png_ptr, info_ptr);

    width = png_get_image_width(png_ptr, info_ptr);
    height = png_get_image_height(png_ptr, info_ptr);
    //color_type = png_get_color_type(png_ptr, info_ptr);
    bit_depth = png_get_bit_depth(png_ptr, info_ptr);

    //number_of_passes = png_set_interlace_handling(png_ptr);
    png_read_update_info(png_ptr, info_ptr);


    /* read file */
    if (setjmp(png_jmpbuf(png_ptr))){
        printf("[read_png_file] Error during read_image\n");
        return image;
    }

    row_pointers = (png_bytep*) malloc(sizeof(png_bytep) * height);
    for (int y=0; y<height; y++) {
        row_pointers[y] = (png_byte *) malloc(png_get_rowbytes(png_ptr, info_ptr));
    }
    png_read_image(png_ptr, row_pointers);

    fclose(fp);

    if (png_get_color_type(png_ptr, info_ptr) == PNG_COLOR_TYPE_RGB){
        image = createImage(width,height,3);
        image->scalingFactor = pow(2,bit_depth)-1;
        image->channelDepth = bit_depth;
        image->colorSpace = RGB;
        for (int y=0; y<height; y++) {
            png_byte* row = row_pointers[y];
            for (int x=0; x<width; x++) {
                png_byte* ptr = &(row[x*3]);

                imageValCh(image,x,y,0) = ptr[0];
                imageValCh(image,x,y,1) = ptr[1];
                imageValCh(image,x,y,2) = ptr[2];
            }
        }

    }else if(png_get_color_type(png_ptr, info_ptr) == PNG_COLOR_TYPE_RGB_ALPHA){
        image = createImage(width,height,4);
        image->scalingFactor = pow(2,bit_depth)-1;
        image->channelDepth = bit_depth;
        image->colorSpace = RGBA;
        for (int y=0; y<height; y++) {
            png_byte* row = row_pointers[y];
            for (int x=0; x<width; x++) {
                png_byte* ptr = &(row[x*4]);
                imageValCh(image,x,y,0) = ptr[0];
                imageValCh(image,x,y,1) = ptr[1];
                imageValCh(image,x,y,2) = ptr[2];
                imageValCh(image,x,y,3) = ptr[3];
            }
        }
    }else if(png_get_color_type(png_ptr, info_ptr) == PNG_COLOR_TYPE_GRAY){
        image = createImage(width,height,1);
        image->scalingFactor = pow(2,bit_depth)-1;
        image->channelDepth = bit_depth;
        image->colorSpace = GRAYSCALE;
        for (int y=0; y<height; y++) {
            png_byte* row = row_pointers[y];
            for (int x=0; x<width; x++) {
                png_byte* ptr = &(row[x*1]);
                imageValCh(image,x,y,0) = ptr[0];
            }
        }
    }else if(png_get_color_type(png_ptr, info_ptr) == PNG_COLOR_TYPE_GRAY_ALPHA){
        image = createImage(width,height,2);
        image->scalingFactor = pow(2,bit_depth)-1;
        image->channelDepth = bit_depth;
        image->colorSpace = GRAYSCALE_ALPHA;
        for (int y=0; y<height; y++) {
            png_byte* row = row_pointers[y];
            for (int x=0; x<width; x++) {
                png_byte* ptr = &(row[x*2]);
                imageValCh(image,x,y,0) = ptr[0];
                imageValCh(image,x,y,1) = ptr[1];
            }
        }
    }else if(png_get_color_type(png_ptr, info_ptr) == PNG_COLOR_TYPE_PALETTE){
        printf("PNG_COLOR_TYPE_PALETTE is not implemented yet\n");
    }else{
        printf("PNG_COLOR_TYPE_ unknown\n");
    }
    for (int y=0; y<height; y++) {
        free(row_pointers[y]);
    }
    free(row_pointers);
    return image;

}

Image *readImageJPEG(char *filename){
    //code from jpeg-9b/example.c
    Image* image = NULL;
    /* This struct contains the JPEG decompression parameters and pointers to
    * working space (which is allocated as needed by the JPEG library).
    */
    struct jpeg_decompress_struct cinfo;
    /* We use our private extension JPEG error handler.
    * Note that this struct must live as long as the main JPEG parameter
    * struct, to avoid dangling-pointer problems.
    */
    struct jpeg_error_mgr jerr;
    //struct my_error_mgr jerr;

    /* More stuff */
    FILE * infile;		/* source file */
    JSAMPARRAY buffer;		/* Output row buffer */
    int row_stride;		/* physical row width in output buffer */
    /* In this example we want to open the input file before doing anything else,
 * so that the setjmp() error recovery below can assume the file is open.
 * VERY IMPORTANT: use "b" option to fopen() if you are on a machine that
 * requires it in order to read binary files.
 */

    if ((infile = fopen(filename, "rb")) == NULL) {
        printf("[readImageJPEG] can't open %s\n",filename);
        return NULL;
    }

    /* Step 1: allocate and initialize JPEG decompression object */

    /* We set up the normal JPEG error routines, then override error_exit. */
    cinfo.err = jpeg_std_error(&jerr);
    //jerr.pub.error_exit = my_error_exit;

    /* Establish the setjmp return context for my_error_exit to use. */
    jmp_buf setjmp_buffer;
    if (setjmp(setjmp_buffer)) {
        /* If we get here, the JPEG code has signaled an error.
         * We need to clean up the JPEG object, close the input file, and return.
         */
        jpeg_destroy_decompress(&cinfo);
        printf("[readImageJPEG] code has signaled an error\n");
        fclose(infile);
        return NULL;
    }
    /* Now we can initialize the JPEG decompression object. */
    jpeg_create_decompress(&cinfo);
    /* Step 2: specify data source (eg, a file) */
    jpeg_stdio_src(&cinfo, infile);
    /* Step 3: read file parameters with jpeg_read_header() */
    (void) jpeg_read_header(&cinfo, TRUE);
    /* We can ignore the return value from jpeg_read_header since
     *   (a) suspension is not possible with the stdio data source, and
     *   (b) we passed TRUE to reject a tables-only JPEG file as an error.
     * See libjpeg.txt for more info.
     */

    /* Step 4: set parameters for decompression */

    /* In this example, we don't need to change any of the defaults set by
     * jpeg_read_header(), so we do nothing here.
     */

    /* Step 5: Start decompressor */
    (void) jpeg_start_decompress(&cinfo);
    /* We can ignore the return value since suspension is not possible
 * with the stdio data source.
 */

    /* We may need to do some setup of our own at this point before reading
     * the data.  After jpeg_start_decompress() we have the correct scaled
     * output image dimensions available, as well as the output colormap
     * if we asked for color quantization.
     * In this example, we need to make an output work buffer of the right size.
     */
    /* JSAMPLEs per row in output buffer */
    row_stride = cinfo.output_width * cinfo.output_components;
    /* Make a one-row-high sample array that will go away when done with image */
    buffer = (*cinfo.mem->alloc_sarray)
            ((j_common_ptr) &cinfo, JPOOL_IMAGE, row_stride, 1);

    /* Step 6: while (scan lines remain to be read) */
    /*           jpeg_read_scanlines(...); */

    /* Here we use the library's state variable cinfo.output_scanline as the
     * loop counter, so that we don't have to keep track ourselves.
     */
    image = createImage(cinfo.image_width,cinfo.image_height,cinfo.num_components);
    image->channelDepth = cinfo.data_precision;
    image->scalingFactor = pow(2,image->channelDepth)-1;

    //printf("%d %d\n",cinfo.jpeg_color_space, cinfo.out_color_space);

    if(cinfo.out_color_space == JCS_GRAYSCALE){
        image->colorSpace = GRAYSCALE;
    }else if(cinfo.out_color_space == JCS_RGB){//maybe a lib bug?
        image->colorSpace = RGB;
    }else if(cinfo.out_color_space == JCS_YCbCr){//maybe a lib bug?
        image->colorSpace = YCbCr;
    }else if(cinfo.out_color_space == JCS_CMYK){
        image->colorSpace = CMYK;
    }else if(cinfo.out_color_space == JCS_YCCK){
        image->colorSpace = CMYK;
//     } else if(cinfo.out_color_space == JCS_BG_RGB){
//         printf("[readImageJPEG] big gamut R/G/B not implemented yet, image color space will be set to RGB\n");
//         image->colorSpace = RGB;
//         //return NULL;
//     }else if(cinfo.out_color_space == JCS_BG_YCC){
//         printf("[readImageJPEG] big gamut Y/Cb/Cr not implemented yet, image color space will be set to YCbCr\n");
//         image->colorSpace = YCbCr;
//         //return NULL;
    }else{
        printf("[readImageJPEG] unknown color space\n");
        image->colorSpace = UNKNOWN;
    }
    unsigned  int imageRow = 0;
    while (cinfo.output_scanline < cinfo.output_height){
        jpeg_read_scanlines(&cinfo, buffer, 1);
        unsigned int imageCol = 0;
        for (unsigned int i = 0; i < (unsigned int)cinfo.output_width; i++) {
            int shift = i*cinfo.num_components;
            for (int c = 0; c < cinfo.num_components; ++c) {
                imageValCh(image,imageCol,imageRow,c) = buffer[0][shift+c];
            }
            imageCol++;
        }
        imageRow++;
    }

    /* Step 7: Finish decompression */
    (void) jpeg_finish_decompress(&cinfo);

    /* This is an important step since it will release a good deal of memory. */
    jpeg_destroy_decompress(&cinfo);

    /* After finish_decompress, we can close the input file.
     * Here we postpone it until after no more JPEG errors are possible,
     * so as to simplify the setjmp error logic above.  (Actually, I don't
     * think that jpeg_destroy can do an error exit, but why assume anything...)
     */

    fclose(infile);
    /* At this point you may want to check to see whether any corrupt-data
     * warnings occurred (test whether jerr.pub.num_warnings is nonzero).
     */

    //jerr.num_warnings; //useful to know about corrupted data
    //printf("%ld\n",jerr.num_warnings);
    return image;

}

void writeImageJPEG(Image *image,char *filename){
    /* This struct contains the JPEG compression parameters and pointers to
 * working space (which is allocated as needed by the JPEG library).
 * It is possible to have several such structures, representing multiple
 * compression/decompression processes, in existence at once.  We refer
 * to any one struct (and its associated working data) as a "JPEG object".
 */
    struct jpeg_compress_struct cinfo;

    /* This struct represents a JPEG error handler.  It is declared separately
 * because applications often want to supply a specialized error handler
 * (see the second half of this file for an example).  But here we just
 * take the easy way out and use the standard error handler, which will
 * print a message on stderr and call exit() if compression fails.
 * Note that this struct must live as long as the main JPEG parameter
 * struct, to avoid dangling-pointer problems.
 */
    struct jpeg_error_mgr jerr;

    /* More stuff */
    FILE * outfile;		/* target file */
    JSAMPARRAY buffer;
        /* Step 1: allocate and initialize JPEG compression object */

    /* We have to set up the error handler first, in case the initialization
     * step fails.  (Unlikely, but it could happen if you are out of memory.)
     * This routine fills in the contents of struct jerr, and returns jerr's
     * address which we place into the link field in cinfo.
     */
    cinfo.err = jpeg_std_error(&jerr);

        /* Now we can initialize the JPEG compression object. */
    jpeg_create_compress(&cinfo);
    /* Step 2: specify data destination (eg, a file) */
    /* Note: steps 2 and 3 can be done in either order. */

    /* Here we use the library-supplied code to send compressed data to a
     * stdio stream.  You can also write your own code to do something else.
     * VERY IMPORTANT: use "b" option to fopen() if you are on a machine that
     * requires it in order to write binary files.
     */
    if ((outfile = fopen(filename, "wb")) == NULL) {
        fprintf(stderr, "can't open %s\n", filename);
        exit(1);
    }
    jpeg_stdio_dest(&cinfo, outfile);

        /* First we supply a description of the input image.
 * Four fields of the cinfo struct must be filled in:
 */
    cinfo.image_width = image->nx; 	/* image width and height, in pixels */
    cinfo.image_height = image->ny;

    if(image->colorSpace == RGB || image->colorSpace == RGBA){
        cinfo.input_components = 3;
        cinfo.in_color_space = JCS_RGB;
        cinfo.jpeg_color_space = JCS_RGB;
    }else if(image->colorSpace == GRAYSCALE || image->colorSpace == GRAYSCALE_ALPHA){
        cinfo.input_components = 1;
        cinfo.in_color_space = JCS_GRAYSCALE;
        cinfo.jpeg_color_space = JCS_GRAYSCALE;
    }else if(image->colorSpace == YCbCr){
        cinfo.input_components = 3;
        cinfo.in_color_space = JCS_YCbCr;
        cinfo.jpeg_color_space = JCS_YCbCr;
    }else if(image->colorSpace == CMYK){
        cinfo.input_components = 4;
        cinfo.in_color_space = JCS_CMYK;
        cinfo.jpeg_color_space = JCS_CMYK;
    }else if(image->colorSpace == YCbCrK){
        cinfo.input_components = 4;
        cinfo.in_color_space = JCS_YCCK;
        cinfo.jpeg_color_space = JCS_YCCK;
    }else{
        cinfo.input_components = image->nchannels;
        cinfo.in_color_space = JCS_UNKNOWN;
        cinfo.jpeg_color_space = JCS_UNKNOWN;
    }

        /* Now use the library's routine to set default compression parameters.
 * (You must set at least cinfo.in_color_space before calling this,
 * since the defaults depend on the source color space.)
 */
    jpeg_set_defaults(&cinfo);

        /* Now you can set any non-default parameters you wish to.
 * Here we just illustrate the use of quality (quantization table) scaling:
 */
    int quality = 100;
    jpeg_set_quality(&cinfo, quality, TRUE /* limit to baseline-JPEG values */);

    /* Step 4: Start compressor */
    /* TRUE ensures that we will write a complete interchange-JPEG file.
     * Pass TRUE unless you are very sure of what you're doing.
     */
    jpeg_start_compress(&cinfo, TRUE);

    /* Step 5: while (scan lines remain to be written) */
    /*           jpeg_write_scanlines(...); */

    /* Here we use the library's state variable cinfo.next_scanline as the
     * loop counter, so that we don't have to keep track ourselves.
     * To keep things simple, we pass one scanline per call; you can pass
     * more if you wish, though.
     */
    int row_stride = cinfo.image_width * cinfo.num_components;
    buffer = (*cinfo.mem->alloc_sarray)
            ((j_common_ptr) &cinfo, JPOOL_IMAGE, row_stride, 1);
    unsigned int imageRow = 0;
    while (cinfo.next_scanline < cinfo.image_height) {
        /* jpeg_write_scanlines expects an array of pointers to scanlines.
         * Here the array is only one element long, but you could pass
         * more than one scanline at a time if that's more convenient.
         */
        unsigned int imageCol = 0;
        for (unsigned int i = 0; i < (unsigned int)cinfo.image_width; i++) {
            int shift = i*cinfo.num_components;
            for (int c = 0; c < cinfo.num_components; ++c) {
                buffer[0][(shift+c)] = (unsigned char)imageValCh(image,imageCol,imageRow,c);
            }
            imageCol++;
        }
        imageRow++;
        (void) jpeg_write_scanlines(&cinfo, buffer, 1);
    }

    /* Step 6: Finish compression */

    jpeg_finish_compress(&cinfo);
    /* After finish_compress, we can close the output file. */
    fclose(outfile);

    /* Step 7: release JPEG compression object */

    /* This is an important step since it will release a good deal of memory. */
    jpeg_destroy_compress(&cinfo);
}

void writeImagePNG(Image *image,char *filename){
/*
 * Copyright 2002-2010 Guillaume Cottenceau.
 *
 * This software may be freely redistributed under the terms
 * of the X11 license.
 *
 */

    /* create file */
    FILE *fp = fopen(filename, "wb");
    if (!fp){
        printf("[write_png_file] File %s could not be opened for writing\n", filename);
    }

    /* initialize stuff */
    png_structp png_ptr;
    png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);

    if (!png_ptr){
        printf("[write_png_file] png_create_write_struct failed\n");
    }

    png_infop info_ptr;
    info_ptr = png_create_info_struct(png_ptr);
    if (!info_ptr){
        printf("[write_png_file] png_create_info_struct failed\n");
    }

    if (setjmp(png_jmpbuf(png_ptr))){
        printf("[write_png_file] Error during init_io\n");
    }

    png_init_io(png_ptr, fp);


    /* write header */
    if (setjmp(png_jmpbuf(png_ptr))){
        printf("[write_png_file] Error during writing header\n");
    }
    int width, height;
    width = image->nx;
    height = image->ny;
    png_byte bit_depth = image->channelDepth;
    png_byte color_type;
    if(image->colorSpace == RGB) {
        color_type = PNG_COLOR_TYPE_RGB;
    } else if(image->colorSpace == RGBA) {
        color_type = PNG_COLOR_TYPE_RGB_ALPHA;
    }else if(image->colorSpace == GRAYSCALE){
        color_type = PNG_COLOR_TYPE_GRAY;
    }else if(image->colorSpace == GRAYSCALE_ALPHA){
        color_type = PNG_COLOR_TYPE_GRAY_ALPHA;
    }else{
        if(image->nchannels == 1){
            color_type = PNG_COLOR_TYPE_GRAY;
        }
        else if(image->nchannels == 2){
            color_type = PNG_COLOR_TYPE_GRAY_ALPHA;
        }
        else if(image->nchannels == 3){
            color_type = PNG_COLOR_TYPE_RGB;
        }
        else if(image->nchannels == 4){
            color_type = PNG_COLOR_TYPE_RGB_ALPHA;
        }else{
            printf("[writeImagePNG] unknown color space\n");
        }
    }

    png_set_IHDR(png_ptr, info_ptr, width, height,
                 bit_depth, color_type, PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_BASE, PNG_FILTER_TYPE_BASE);

    png_write_info(png_ptr, info_ptr);


    /* write bytes */
    if (setjmp(png_jmpbuf(png_ptr))){
        printf("[write_png_file] Error during writing bytes\n");
    }

    png_bytep * row_pointers;
    row_pointers = (png_bytep*) malloc(sizeof(png_bytep) * height);
    for (int y=0; y<height; y++) {
        row_pointers[y] = (png_byte *) malloc(png_get_rowbytes(png_ptr, info_ptr));
    }

    if(image->colorSpace == RGB) {
        for (int y=0; y<height; y++) {
            png_byte* row = row_pointers[y];
            for (int x=0; x<width; x++) {
                png_byte* ptr = &(row[x*3]);
                ptr[0]  = (png_byte)(imageValCh(image,x,y,0));
                ptr[1] =  (png_byte)(imageValCh(image,x,y,1));
                ptr[2] =  (png_byte)(imageValCh(image,x,y,2));
            }
        }
    } else if(image->colorSpace == RGBA) {
        for (int y=0; y<height; y++) {
            png_byte* row = row_pointers[y];
            for (int x=0; x<width; x++) {
                png_byte* ptr = &(row[x*4]);
                ptr[0]  = (png_byte)(imageValCh(image,x,y,0));
                ptr[1] =  (png_byte)(imageValCh(image,x,y,1));
                ptr[2] =  (png_byte)(imageValCh(image,x,y,2));
                ptr[3] =  (png_byte)(imageValCh(image,x,y,3));
            }
        }
    }else if(image->colorSpace == GRAYSCALE){
        for (int y=0; y<height; y++) {
            png_byte* row = row_pointers[y];
            for (int x=0; x<width; x++) {
                png_byte* ptr = &(row[x*1]);
                ptr[0]  = (png_byte)(imageValCh(image,x,y,0));
            }
        }
    }else if(image->colorSpace == GRAYSCALE_ALPHA){
        for (int y=0; y<height; y++) {
            png_byte* row = row_pointers[y];
            for (int x=0; x<width; x++) {
                png_byte* ptr = &(row[x*2]);
                ptr[0]  = (png_byte)(imageValCh(image,x,y,0));
                ptr[1]  = (png_byte)(imageValCh(image,x,y,1));
            }
        }
    }else{
        printf("unknown color type\n");
    }

    png_write_image(png_ptr, row_pointers);


    /* end write */
    if (setjmp(png_jmpbuf(png_ptr))){
        printf("[write_png_file] Error during end of write");
    }

    png_write_end(png_ptr, NULL);

    /* cleanup heap allocation */
    for (int y=0; y<height; y++){
        free(row_pointers[y]);
    }
    free(row_pointers);

    fclose(fp);
}

void writeImagePGM(Image *image,char *filename, char *magicNumber){

    if (strcmp(magicNumber,"P2")==0){
        writeImageP2(image,filename);
    }
    else if(strcmp(magicNumber,"P5")==0){
        writeImageP5(image,filename);
    }else{
        printf("unknown magic number");
    }
}

void writeImagePPM(Image *image,char *filename, char *magicNumber){

    if (strcmp(magicNumber,"P3")==0){
        writeImageP3(image,filename);
    }
    else if(strcmp(magicNumber,"P6")==0){
        writeImageP6(image,filename);
    }else{
        printf("unknown magic number");
    }
}


void writeImageP2(Image *image,char *filename){
    int i, j;
    FILE *fp = fopen(filename, "w");
    (void) fprintf(fp, "P2\n%d %d\n%d\n", image->nx,image->ny, image->scalingFactor);
    int k = 0;


    float* channel0 = image->channel[0];
    for (i = 0; i < image->ny; i++){
        for (j = 0; j < image->nx; j++){
            (void) fprintf(fp,"%d ",(int)channel0[k]);
            k++;
        }
        (void) fprintf(fp,"\n");
    }


    (void) fclose(fp);
}

void writeImageP3(Image *image,char *filename){
    int i, j;
    FILE *fp = fopen(filename, "w");
    (void) fprintf(fp, "P3\n%d %d\n%d\n", image->nx,image->ny, image->scalingFactor);
    int k=0;

    if(image->dataTypeId == FLOAT){
        float* channel0 = (float*)image->channel[0];
        float* channel1 = (float*)image->channel[1];
        float* channel2 = (float*)image->channel[2];
        for (i = 0; i < image->ny; i++){
            for (j = 0; j < image->nx; j++){
                (void) fprintf(fp,"%d %d %d ", (int)(channel0[k]),(int)channel1[k],(int)channel2[k]);
                k++;
            }
            (void) fprintf(fp,"\n");
        }
    }
    if(image->dataTypeId == DOUBLE){
        double* channel0 = (double*)image->channel[0];
        double* channel1 = (double*)image->channel[1];
        double* channel2 = (double*)image->channel[2];
        for (i = 0; i < image->ny; i++){
            for (j = 0; j < image->nx; j++){
                (void) fprintf(fp,"%d %d %d ", (int)(channel0[k]),(int)channel1[k],(int)channel2[k]);
                k++;
            }
            (void) fprintf(fp,"\n");
        }
    }
    (void) fclose(fp);

}

void writeImageP5(Image *image,char *filename){
    FILE *fp = fopen(filename, "wb");
    (void) fprintf(fp, "P5\n%d %d\n%d\n", image->nx,image->ny, image->scalingFactor);
    int k = 0;

    float* channel0 = image->channel[0];
    for (int i = 0; i < image->ny; i++){
        for (int j = 0; j < image->nx; j++){
            static unsigned char gray[1];
            gray[0] = (int)channel0[k];
            (void) fwrite(gray, 1, 1, fp);
            k++;
        }
    }

    (void) fclose(fp);
}


void writeImageP6(Image *image,char *filename){
    int i, j;
    FILE *fp = fopen(filename, "wb");
    (void) fprintf(fp, "P6\n%d %d\n%d\n", image->nx,image->ny, image->scalingFactor);
    int k=0;

    float* channel0 = image->channel[0];
    float* channel1 = image->channel[1];
    float* channel2 = image->channel[2];
    for (i = 0; i < image->ny; i++){
        for (j = 0; j < image->nx; j++){
            static unsigned char color[3];
            color[0] = channel0[k];
            color[1] = channel1[k];
            color[2] = channel2[k];
            (void) fwrite(color, 1, 3, fp);
            k++;
        }
    }

    (void) fclose(fp);
}

void writeImage(Image* image,char *filename){
    char ext[10],*pos;
    pos = strrchr(filename,'.') + 1;
    sscanf(pos,"%s",ext);

    if (strcmp(ext,"pgm")==0){
        writeImagePGM(image,filename,"P5");
    }
    else if (strcmp(ext,"ppm")==0){
        writeImagePPM(image,filename,"P6");
    }else if(strcmp(ext,"png")==0){
        writeImagePNG(image,filename);
    }else if(strcmp(ext,"jpg")==0){
        writeImageJPEG(image,filename);
    } else if(strcmp(ext,"jpeg")==0){
        writeImageJPEG(image,filename);
    }
    else{
        printf("[writeImage] unsuported format\n");
    }
}



Image *convertRGBtoYCbCr(Image *rgbImage)
{
    Image *ycbcrImage;
    float a = 16.0;
    float b = 128.0;

    ycbcrImage = createImage(rgbImage->nx,rgbImage->ny,3);
    ycbcrImage->scalingFactor = rgbImage->scalingFactor;
    for (int k=0; k < rgbImage->numberPixels; k++){
        float lum  = (0.257*rgbImage->channel[0][k])+
                     (0.504*rgbImage->channel[1][k])+
                     (0.098*rgbImage->channel[2][k])+a;
        float Cb = (-0.148*rgbImage->channel[0][k]) +
                   (-0.291*rgbImage->channel[1][k]) +
                   (0.439*rgbImage->channel[2][k])+b;
        float Cr = (0.439*rgbImage->channel[0][k]) +
                   (-0.368*rgbImage->channel[1][k]) +
                   (-0.071*rgbImage->channel[2][k]) +b;

        if (lum < 0)   lum = 0.0;
        if (lum > ycbcrImage->scalingFactor) lum = ycbcrImage->scalingFactor;
        if (Cb < 0)   Cb = 0.0;
        if (Cb > ycbcrImage->scalingFactor) Cb = ycbcrImage->scalingFactor;
        if (Cr < 0)   Cr = 0.0;
        if (Cr > ycbcrImage->scalingFactor) Cr = ycbcrImage->scalingFactor;

        ycbcrImage->channel[0][k] = lum;
        ycbcrImage->channel[1][k] = Cb;
        ycbcrImage->channel[2][k] = Cr;
    }

    return(ycbcrImage);
}

GrayImage* convertImage2GrayImage(Image* image){
    GrayImage* grayImage = createGrayImage(image->nx,image->ny);
    for (int k = 0; k < image->numberPixels; ++k) {
        grayImage->val[k] = image->channel[0][k];
    }
    return grayImage;
}

Image* convertGrayImage2Image(GrayImage* grayImage){
    Image* image = createImage(grayImage->ncols,grayImage->nrows,1);
    int maxValue = maximumValue(grayImage);
    if(maxValue < 256){
        image->scalingFactor = 255;
    }else{
        int value = 256;
        while(value-1 < maxValue){
            value = value << 1;
        }
        image->scalingFactor = value-1;
    }

    for (int k = 0; k < image->numberPixels; ++k) {
        image->channel[0][k] = grayImage->val[k];
    }
    return image;
}

ColorImage* convertImage2ColorImage(Image* image){
    ColorImage* colorImage = createColorImage(image->nx,image->ny);
    int k = 0;
    for (int y = 0; y < colorImage->ny; ++y) {
        for (int x = 0; x < colorImage->nx; ++x) {
            colorImage->cor[y][x].val[0] = image->channel[0][k];
            colorImage->cor[y][x].val[1] = image->channel[1][k];
            colorImage->cor[y][x].val[2] = image->channel[2][k];
            k++;
        }
    }
    return colorImage;
}

Image* convertColorImage2Image(ColorImage* colorImage){
    Image* image = createImage(colorImage->nx,colorImage->ny,3);
    int k = 0;
    int maxValue = maximumColorValue(colorImage);
    if(maxValue < 256){
        image->scalingFactor = 255;
    }else{
        int value = 256;
        while(value-1 < maxValue){
            value = value << 1;
        }
        image->scalingFactor = value-1;
    }
    for (int y = 0; y < colorImage->ny; ++y) {
        for (int x = 0; x < colorImage->nx; ++x) {
            image->channel[0][k] = colorImage->cor[y][x].val[0];
            image->channel[1][k] = colorImage->cor[y][x].val[1];
            image->channel[2][k] = colorImage->cor[y][x].val[2];
            k++;
        }
    }
    return image;
}

float sumUpAllPixelsValues(Image *image, bool normalize){
    double sum = 0;
    for (int k = 0; k < image->numberPixels; ++k) {
        for (int c = 0; c < image->nchannels; ++c) {
            sum += image->channel[c][k];
        }
    }
    if(normalize){
        sum /= ( image->numberPixels * image->nchannels);
    }
    return ((float)sum);
}

Image* extractImageChannel(Image* image, int channel){
    Image* channelImage = createImage(image->nx,image->ny,1);
    channelImage->scalingFactor = image->scalingFactor;
    for (int k = 0; k < image->numberPixels; ++k) {
        channelImage->channel[0][k] = image->channel[channel][k];
    }
    return channelImage;
}

void printImage(Image* image){
    int index=0;
    for (int y = 0; y < image->ny; ++y) {
        for (int x = 0; x < image->nx; ++x) {
            for (int c = 0; c < image->nchannels; ++c) {
                index = (y*image->nx)+x;
                printf("%.2f ",image->channel[c][index]);
            }
        }
        printf("\n");
    }
}

Image* createAlphaChannel(Image* image,float alpha){
    Image *output = createImage(image->nx,image->ny,image->nchannels+1);
    output->scalingFactor = image->scalingFactor;
    for (int k = 0; k < image->numberPixels; ++k) {
        for (int c = 0; c < image->nchannels; ++c) {
            output->channel[c][k] = image->channel[c][k];
        }
        output->channel[image->nchannels][k] = alpha;
    }
    return output;
}

void addUniformNoise(Image* image, float uniformValue, double probability){
    for (int k = 0; k < image->numberPixels; ++k) {
        if(randomNormalized() <=  probability){
            for (int c = 0; c < image->nchannels; ++c) {
                image->channel[c][k] += uniformValue;
                if(image->channel[c][k] > image->scalingFactor){
                    image->channel[c][k] = image->scalingFactor;
                }
            }
        }
    }
}

void addSaltAndPepperNoise(Image* image, double probability){
    for (int k = 0; k < image->numberPixels; ++k) {
        if(randomNormalized() <=  probability){
            for (int c = 0; c < image->nchannels; ++c) {
                //salt
                if(randomNormalized() >= 0.5){
                    image->channel[c][k] = image->scalingFactor;
                }else{
                    image->channel[c][k] = 0;
                }
            }
        }

    }
}

void addAdditiveGaussianNoise(Image* image, double mean, double variance){


    double standardDeviation = sqrt(variance);
    for (int y = 0; y < image->ny; ++y) {
        for (int x = 0; x < image->nx; ++x) {
            double value = generateGaussianNoise(0,standardDeviation);
            value = value*image->scalingFactor;
            imageVal(image,x,y) += value;
            if(imageVal(image,x,y) < 0){
                imageVal(image,x,y) = 0;
            }
            if(imageVal(image,x,y) > image->scalingFactor){
                imageVal(image,x,y) = image->scalingFactor;
            }
        }
    }
}

uint8_t* convertImage2IntergerArray8bits(Image* imageRGBA){
    uint8_t* output = (uint8_t*)calloc(imageRGBA->nchannels*imageRGBA->nx*imageRGBA->ny ,sizeof(uint8_t));
    long index = 0;
    for (int k = 0; k < imageRGBA->numberPixels; ++k) {
        for (int c = 0; c < imageRGBA->nchannels; ++c) {
            output[index] = (uint8_t)imageRGBA->channel[c][k];
            index++;
        }
    }
    return output;
}

Image* copyImage(Image* image, bool copyChannels){
    Image* imageCopy = createImage(image->nx,image->ny,image->nz,image->nchannels);
    imageCopy->colorSpace = image->colorSpace;
    imageCopy->dx = image->dx;
    imageCopy->dy = image->dy;
    imageCopy->dz = image->dz;
    imageCopy->dataTypeId = image->dataTypeId;
    imageCopy->scalingFactor = image->scalingFactor;
    for (int i = 0; i < 10; ++i) {
        imageCopy->unid[i] = image->unid[i];
    }
    if(copyChannels) {
        for (int k = 0; k < image->numberPixels; ++k) {
            for (int c = 0; c < image->nchannels; ++c) {
                imageCopy->channel[c][k] = image->channel[c][k];
            }
        }
    }
    return imageCopy;
}

Image *imageSubtraction(Image *image1, Image *image2, bool saturation){
    Image *outputImage = NULL;
    if(isImagesSameDomain(image1,image2)){
        outputImage = createImage(image1->nx,image1->ny,image1->nchannels);
        int n = image1->nx*image1->ny;
        int result;
        for (int i = 0; i < n; ++i) {
            for(int c=0; c<image1->nchannels; c++){
                result = image1->channel[c][i] - image2->channel[c][i];
                if(saturation){
                    result = (result<0)?0:result;
                }
                outputImage->channel[c][i] = result;
            }

        }
    }
    return outputImage;
}

Image* packImagesFromDirectory(DirectoryManager* directoryManager){
    Image* imagePack = NULL;
    Image* currentImage = NULL;
    Image* merged = NULL;
    for (int i = 0; i < (int)directoryManager->nfiles; ++i) {
        currentImage = readImage(directoryManager->files[i]->path);
        merged = mergeImages(imagePack,currentImage);
        destroyImage(&currentImage);
        destroyImage(&imagePack);
        imagePack = copyImage(merged,true);
        destroyImage(&merged);
    }
    return imagePack;
}

Image* convertGrayImage2RGBImage(Image* image_ppm){
    Image* image_rgb = createImage(image_ppm->nx,image_ppm->ny,3);
    for (int k = 0; k < image_rgb->numberPixels; ++k) {
        image_rgb->channel[0][k] = image_ppm->channel[0][k];
        image_rgb->channel[1][k] = image_ppm->channel[0][k];
        image_rgb->channel[2][k] = image_ppm->channel[0][k];
    }
    return image_rgb;
}

Image* mergeImages(Image* image1, Image* image2){
    Image* imageMerged = NULL;
    if(image1 == NULL && image2 == NULL){
        printf("NULL images\n");
        return NULL;
    }
    if(image1 == NULL){
        imageMerged = copyImage(image2,true);
        return imageMerged;
    }
    if(image2 == NULL){
        imageMerged = copyImage(image1,true);
        return imageMerged;
    }

    if(image1->nx != image2->nx || image1->ny != image2->ny){
        printf("images dimensions mismatch\n");
        return NULL;
    }
    if(image1->nchannels != image2->nchannels){
        printf("number of channels mismatch\n");
        return NULL;
    }
    int totalSlices = image1->nz+image2->nz;
    imageMerged = createImage(image1->nx,image1->ny,totalSlices,image1->nchannels);

#pragma omp parallel for
    for (int k = 0; k < image1->numberPixels; ++k) {
        for (int c = 0; c < image1->nchannels; ++c) {
            imageMerged->channel[c][k] = image1->channel[c][k];
        }
    }

    for (int k = 0,kShift = image1->numberPixels; k < image2->numberPixels; ++k,++kShift) {
        for (int c = 0; c < image2->nchannels; ++c) {
            imageMerged->channel[c][kShift] = image2->channel[c][k];
        }
    }
    return imageMerged;
}

Image* getSlice(Image* image, int sliceIndex){
    if(sliceIndex >= image->nz || sliceIndex < 0){
        printf("invalid sliceIndex (z-coordinate) value\n");
        return NULL;
    }
    Image* slice = createImage(image->nx,image->ny,image->nchannels);
    for (int y = 0; y < image->ny; ++y) {
        for (int x = 0; x < image->nx; ++x) {
            for (int c = 0; c < image->nchannels; ++c) {
                imageValCh(slice,x,y,c) = imageVolumeCh(image,x,y,sliceIndex,c);
            }
        }
    }
    return slice;
}

Image* extractSubImage(Image*image, int xCoord,int yCoord, int xsize, int ysize, bool zeroPadding){
    Image *subImage = NULL;
    int imageX = xCoord+xsize;
    int imageY = yCoord+ysize;
    int subImageSizeX = 0;
    int subImageSizeY = 0;

    subImageSizeX = xsize;
    subImageSizeY = ysize;

    if(image->nx <= xCoord && image->ny <= yCoord){
        printf("Warning: xCoord and yCoord are outbound.\n");
    }

    if(image->nx < imageX){
        if(!zeroPadding){
            subImageSizeX = image->nx-xCoord;
            if(subImageSizeX <= 0){
                printf("Impossible create a patch without zero padding\n");
                return NULL;
            }
        }
    }
    if(image->ny < imageY){
        if(!zeroPadding){
            subImageSizeY = image->ny-yCoord;
            if(subImageSizeY <= 0){
                printf("Impossible create a patch without zero padding\n");
                return NULL;
            }
        }
    }

    subImage = createImage(subImageSizeX,subImageSizeY,image->nchannels);
    for (int y = 0; y < subImage->ny; ++y) {
        for (int x = 0; x < subImage->nx; ++x) {
            for (int c = 0; c < subImage->nchannels; ++c) {
                if(isValidPixelCoordinate(image,xCoord+x,yCoord+y)){
                    imageValCh(subImage,x,y,c) = imageValCh(image,(xCoord+x),(yCoord+y),c);
                }
            }
        }
    }
    return subImage;
}