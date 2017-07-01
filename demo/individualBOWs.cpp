#include "FL.h"
#include <vector>

int main(int argc, char **argv) {
    size_t numberOfVisualWords = 1000;


    // Datasets
//     char const* const fileName_createDict = "/home/akira-miasato/data/img/coil100_train.txt";
//     char const* const fileName_createTrain = "/home/akira-miasato/data/img/coil100_train.txt";
//     char const* const fileName_createTest = "/home/akira-miasato/data/img/coil100_dev.txt";
//     size_t color_patch = 64;
//     size_t grad_patch = 64;
    
//     char const* const fileName_createDict = "/home/akira-miasato/data/img/pubfig_train.txt";
//     char const* const fileName_createTrain = "/home/akira-miasato/data/img/pubfig_train.txt";
//     char const* const fileName_createTest = "/home/akira-miasato/data/img/pubfig_dev.txt";
//     size_t color_patch = 25;
//     size_t grad_patch = 50;
    
    char const* const fileName_createDict = "/home/akira-miasato/data/img/corel_train.txt";
    char const* const fileName_createTrain = "/home/akira-miasato/data/img/corel_train.txt";
    char const* const fileName_createTest= "/home/akira-miasato/data/img/corel_dev.txt";
    size_t color_patch = 29;
    size_t grad_patch = 29;    

//     char const* const fileName_createDict = "/home/akira-miasato/data/img/mnist_train.txt";
//     char const* const fileName_createTrain = "/home/akira-miasato/data/img/mnist_train.txt";
//     char const* const fileName_createTest = "/home/akira-miasato/data/img/mnist_dev.txt";

    
    
    GVector* vectorSamplesUsed2CreateDict =  splitsLinesInTextFile(fileName_createDict);
    GVector* vectorSamplesUsed2TrainClassifier =  splitsLinesInTextFile(fileName_createTrain);
    GVector* vectorSamplesUsed2TestClassifier =  splitsLinesInTextFile(fileName_createTest);
    if(vectorSamplesUsed2CreateDict->size == 0){
        printf("no path found");
        return -1;
    }
    if(vectorSamplesUsed2TrainClassifier->size == 0){
        printf("no path found");
        return -1;
    }
    if(vectorSamplesUsed2TestClassifier->size == 0){
        printf("no path found");
        return -1;
    }
    std::vector<BagOfVisualWordsManager*> managers;

    // Color Hist Bow Manager
    BagOfVisualWordsManager* colorBow = createBagOfVisualWordsManager();
    colorBow->pathsToImages_dictionary = vectorSamplesUsed2CreateDict;//criar o dicionario
    colorBow->pathsToImages_train = vectorSamplesUsed2TrainClassifier;//treinar o classificador
    colorBow->pathsToImages_test = vectorSamplesUsed2TestClassifier;//testar o classificador
    /////////////////////////////////////////////////////////////////////////////
    colorBow->imageSamplerFunction = iftSamplingBow;
    ArgumentList* args = createArgumentList();
    ARGLIST_PUSH_BACK_AS(size_t,args,color_patch); //patch size X
    ARGLIST_PUSH_BACK_AS(size_t,args,color_patch); //patch size Y
    colorBow->argumentListOfSampler = args;//passando a lista de argumentos para o bow manager
    colorBow->freeFunction2SamplerOutput = destroyImageVoidPointer;
    /////////////////////////////////////////////////////////////////////////////
    colorBow->featureExtractorFunction = computeColorHistogramBow;//ponteiro da funcao para a extracao de features
    ArgumentList* colorFeatureExtractorArguments = createArgumentList();
    size_t nbins = 4;
    ARGLIST_PUSH_BACK_AS(size_t,colorFeatureExtractorArguments,nbins); //nBins per channel
    ARGLIST_PUSH_BACK_AS(size_t,colorFeatureExtractorArguments,nbins*nbins*nbins); //total number of channels
    colorBow->argumentListOfFeatureExtractor = colorFeatureExtractorArguments; //passando a lista de argumentos do feature extractor para o bow manager
    /////////////////////////////////////////////////////////////////////////////
    colorBow->distanceFunction = computeNormalizedL1Norm;
    colorBow->argumentListOfDistanceFunction = NULL;
    /////////////////////////////////////////////////////////////////////////////
    colorBow->clusteringFunction = kmeansClusteringBow;
    ArgumentList* clusteringMethodArguments = createArgumentList();
    ARGLIST_PUSH_BACK_AS(size_t,clusteringMethodArguments,numberOfVisualWords); //number of words
    ARGLIST_PUSH_BACK_AS(size_t,clusteringMethodArguments,100); //maximum number of iterations
    ARGLIST_PUSH_BACK_AS(double,clusteringMethodArguments, 1e-4); //tolerance
    ARGLIST_PUSH_BACK_AS(int,clusteringMethodArguments,0); //seed
    ARGLIST_PUSH_BACK_AS(DistanceFunction,clusteringMethodArguments,computeNormalizedL1Norm); //seed
    ARGLIST_PUSH_BACK_AS(ArgumentList*,clusteringMethodArguments,NULL); //seed
    colorBow->argumentListOfClustering = clusteringMethodArguments;
    /////////////////////////////////////////////////////////////////////////////
    computeDictionary(colorBow);
    /////////////////////////////////////////////////////////////////////////////
    colorBow->mountHistogramFunction = computeCountHistogram_bow;
    colorBow->argumentListOfHistogramMounter = NULL;
    ///////////////////////
    managers.push_back(colorBow);

    
    // HOG Bow Manager
    BagOfVisualWordsManager* hogBow = createBagOfVisualWordsManager();
    hogBow->pathsToImages_dictionary = vectorSamplesUsed2CreateDict;//criar o dicionario
    hogBow->pathsToImages_train = vectorSamplesUsed2TrainClassifier;//treinar o classificador
    hogBow->pathsToImages_test = vectorSamplesUsed2TestClassifier;//testar o classificador
    /////////////////////////////////////////////////////////////////////////////
    hogBow->imageSamplerFunction = randomSamplingBow;
    args = createArgumentList();
    ARGLIST_PUSH_BACK_AS(size_t,args,grad_patch); //patch size X
    ARGLIST_PUSH_BACK_AS(size_t,args,grad_patch); //patch size Y
    ARGLIST_PUSH_BACK_AS(size_t,args,20); //patch size Y
//     ARGLIST_PUSH_BACK_AS(size_t,args,20);
    hogBow->argumentListOfSampler = args;//passando a lista de argumentos para o bow manager
    hogBow->freeFunction2SamplerOutput = destroyImageVoidPointer;
    /////////////////////////////////////////////////////////////////////////////
    hogBow->featureExtractorFunction = computeHOGBow;//ponteiro da funcao para a extracao de features
    ArgumentList* hogArguments = createArgumentList();
    nbins = 18;
    ARGLIST_PUSH_BACK_AS(size_t, hogArguments, nbins);
    hogBow->argumentListOfFeatureExtractor = hogArguments;
    hogBow->argumentListOfFeatureExtractor = colorFeatureExtractorArguments; //passando a lista de argumentos do feature extractor para o bow manager
    /////////////////////////////////////////////////////////////////////////////
    hogBow->distanceFunction = computeNormalizedL1Norm;
    hogBow->argumentListOfDistanceFunction = NULL;
    /////////////////////////////////////////////////////////////////////////////
    hogBow->clusteringFunction = kmeansClusteringBow;
    args = createArgumentList();
    ARGLIST_PUSH_BACK_AS(size_t,args,numberOfVisualWords); //number of words
    ARGLIST_PUSH_BACK_AS(size_t,args,100); //maximum number of iterations
    ARGLIST_PUSH_BACK_AS(double,args, 1e-4); //tolerance
    ARGLIST_PUSH_BACK_AS(int,args,0); //seed
    ARGLIST_PUSH_BACK_AS(DistanceFunction,args,computeNormalizedL1Norm); //seed
    ARGLIST_PUSH_BACK_AS(ArgumentList*,args,NULL); //seed
    hogBow->argumentListOfClustering = args;
    /////////////////////////////////////////////////////////////////////////////
    computeDictionary(hogBow);
    /////////////////////////////////////////////////////////////////////////////
    hogBow->mountHistogramFunction = computeCountHistogram_bow;
    hogBow->argumentListOfHistogramMounter = NULL;
    ///////////////////////
    managers.push_back(hogBow);


    //SVM Classifier
    SVM_Classifier* classifiersvm = createSVMClassifier();
    classifiersvm->param.kernel_type = RBF;
    classifiersvm->param.gamma = 3.5;
    managers[0]->classifier = (void*)classifiersvm;
    managers[0]->fitFunction = svm_Classifier_fit;
    managers[0]->storeTrainData = false;
    managers[0]->predictFunction = svm_Classifier_predict;
    managers[0]->storePredictedData = false;
    managers[0]->freeFunctionClassifier = destroySVMClassifierForVoidPointer;
    //////////////////////////////////////

    ///////
    //monta os histogramas, le os label e em seguida treina o classificador
    trainClassifier(managers);
    //////////

    /////////////////////////////////////////////////////
    //monta os histogramas e usa o classificador treinado para
    //classificar as amostras do conjunto de teste
    GVector* labelsPredicted = predictLabels(managers);
    //////////////////////////

    //////////////////////////
    //Le os true labels das imagens e checa com os labels predizidos pelo o classificador.
    //computa uma simples acuracia (numero de amostras rotuladas corretamente / numero de amostras do conjunto)
    GVector* trueLabels = createNullVector(managers[0]->pathsToImages_test->size,sizeof(int));
    int hit = 0;
    printf("file | predicted true\t\tcorrect\n");
    char symbol;
    for (size_t index = 0; index < managers[0]->pathsToImages_test->size; ++index) {
        symbol = 'X';
        char * path = VECTOR_GET_ELEMENT_AS(char*,managers[0]->pathsToImages_test,index);
        VECTOR_GET_ELEMENT_AS(int,trueLabels,index) = findTrueLabelInName(path);
        if(VECTOR_GET_ELEMENT_AS(int,trueLabels,index) == VECTOR_GET_ELEMENT_AS(int,labelsPredicted,index)){
            hit++;
            symbol = 'O';
        }
        printf("%s | %d %d\t\t%c\n",
               path,
               VECTOR_GET_ELEMENT_AS(int,labelsPredicted,index),
               VECTOR_GET_ELEMENT_AS(int,trueLabels,index),symbol
        );
    }
    double acuracia = ((double)hit)/managers[0]->pathsToImages_test->size;
    printf("acuracia: %f\n",acuracia);
    /////////////////////////////////////
    for(BagOfVisualWordsManager* manager : managers){
        destroyBagOfVisualWordsManager(&manager);
    }
    destroyVector(&trueLabels);
    destroyVector(&labelsPredicted);
    return 0;

}



