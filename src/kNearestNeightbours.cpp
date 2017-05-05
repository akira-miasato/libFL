#include "kNearestNeighbours.h"

int moda(std::vector<std::pair<float, int> > vec){
    int max = 0;
    for(int i=0; i<vec.size(); i++){
        if(vec[i].second > max){
            max = vec[i].second;
        }
    }
    max++;
    std::vector<int> n(max);
    for(int i=0; i<max; i++){
        n[i] = 0;
    }
    for(int i=0; i<vec.size(); i++){
        n[vec[i].second]++;
    }
    int ret = 0;
    int argret = 0;
    for(int i=0; i<max; i++){
        if(n[i] > argret){
            ret = i;
            argret = n[i];
        }
    }
    return ret;
}

bool comp_tuple(std::pair<float, int> i,
                std::pair<float, int> j){
    return(i.first < j.first);
}


int nearest(FeatureVector* ref, FeatureMatrix* hyps, VectorDistFn distFn){
    float d, di;
    std::vector<std::pair<float, int> > dist_pos(hyps->nFeaturesVectors);
    d = std::numeric_limits<float>::max();
    for(int j=0; j<hyps->nFeaturesVectors; j++){
        di = distFn(hyps->featureVector[j], ref);
        dist_pos[j].first = di;
        dist_pos[j].second = j;
    }
    std::sort(dist_pos.begin(), dist_pos.end(), comp_tuple);
    return dist_pos[0].second;
}


std::vector<int> knn(FeatureMatrix* target, FeatureMatrix* trainX,
                     std::vector<int> trainY,
                     int k,
                     VectorDistFn distFn
                    ){
    if(trainX->nFeaturesVectors != trainY.size()){
        throw std::runtime_error("X and Y from train ref mismatch!\n");
    }
    std::vector<int> ret(target->nFeaturesVectors);
    std::vector<std::pair<float, int> > dist_pos(trainX->nFeaturesVectors);
    float d, di;
    for(int i=0; i<target->nFeaturesVectors; i++){
        d = std::numeric_limits<float>::max();
        for(int j=0; j<trainX->nFeaturesVectors; j++){
            di = distFn(
              trainX->featureVector[j],
              target->featureVector[i]
            );
            dist_pos[j].first = di;
            dist_pos[j].second = trainY[j];
        }
        std::sort(dist_pos.begin(), dist_pos.end(), comp_tuple);
        std::vector<std::pair<float, int> > knearest(
            dist_pos.begin(), dist_pos.begin() + k  
        );
        ret[i] = moda(knearest);
    }
    return ret;
}
