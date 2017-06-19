#include "iftPoi.h"

iftImage *iftCompute_ISF_MIX_MEAN_Superpixels(iftImage *img, int nsuperpixels, float alpha, float beta, int niters, int smooth_niters, int *nseeds, int *finalniters) {
  iftImage  *mask1, *seeds, *label;
  iftMImage *mimg;
  iftAdjRel *A;
  iftIGraph *igraph;

  /* Set adjacency relation */
  if (iftIs3DImage(img)){
    A      = iftSpheric(1.0);
  } else {
    A      = iftCircular(1.0);
  }

  
  if (iftIsColorImage(img)){
    /* RGB to Lab conversion */
    mimg   = iftImageToMImage(img,LABNorm_CSPACE);
  } else {
    mimg   = iftImageToMImage(img,GRAY_CSPACE);
  }

  mask1  = iftSelectImageDomain(mimg->xsize,mimg->ysize,mimg->zsize);

  /* Minima of a basins manifold in that domain */
  igraph = iftImplicitIGraph(mimg,mask1,A);
  
  /* Seed sampling for ISF */
  seeds   = iftAltMixedSampling(mimg,mask1,nsuperpixels);

  *nseeds = iftNumberOfElements(seeds);

  iftDestroyImage(&mask1);
  iftDestroyMImage(&mimg);

  /* Superpixel segmentation */
  *finalniters = iftIGraphISF_Mean(igraph,seeds,alpha,beta,niters);

  /* Smooth regions in the label map of igraph */  
  if (smooth_niters > 0){
    iftIGraphSetWeightForRegionSmoothing(igraph, img);
    iftIGraphSmoothRegions(igraph, smooth_niters);
  }
  /* Get superpixel image */
  label   = iftIGraphLabel(igraph);

  iftDestroyImage(&seeds);
  iftDestroyIGraph(&igraph);
  iftDestroyAdjRel(&A);
  

  return label;
}

