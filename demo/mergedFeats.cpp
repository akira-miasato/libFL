#include "FL.h"
#include <vector>

int main(int argc, char **argv) {
    size_t numberOfVisualWords = 1000;


    //Caminhos onde esta o arquivo txt gerado pelo o script python "selec_samples2.py"
    //os caminhos vao mudar para cada pessoa
    char const* const fileName_createDict = "/home/akira-miasato/data/img/coil100_train.txt";
    char const* const fileName_createTrain = "/home/akira-miasato/data/img/coil100_train.txt";
    char const* const fileName_createTest = "/home/akira-miasato/data/img/coil100_dev.txt";

//     char const* const fileName_createDict = "/home/akira-miasato/data/img/pubfig_train.txt";
//     char const* const fileName_createTrain = "/home/akira-miasato/data/img/pubfig_train.txt";
//     char const* const fileName_createTest = "/home/akira-miasato/data/img/pubfig_dev.txt";

//     char const* const fileName_createDict = "/home/akira-miasato/data/img/mnist_train.txt";
//     char const* const fileName_createTrain = "/home/akira-miasato/data/img/mnist_train.txt";
//     char const* const fileName_createTest = "/home/akira-miasato/data/img/mnist_dev.txt";
    
    //cada posicao do vetor tem uma string para o caminho de uma imagem
    GVector* vectorSamplesUsed2CreateDict =  splitsLinesInTextFile(fileName_createDict);
    GVector* vectorSamplesUsed2TrainClassifier =  splitsLinesInTextFile(fileName_createTrain);
    GVector* vectorSamplesUsed2TestClassifier =  splitsLinesInTextFile(fileName_createTest);

    //apenas checkando se o vetor vazio. Caso o vetor esteja vazio, talvez seu caminho ate o arquivo
    //txt nao esteja correto
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

    //pipeline para a construncao do dicionario. Para mais detalhes olhe a imagem que esta
    //em data/bowArquiteturaImplementada.png

    //a estrutura bow manager e encarregada de fazer o processo do bow
    BagOfVisualWordsManager* bowManager = createBagOfVisualWordsManager();
    ////////////////////////////////////////////////////////////////////////
    //Passando os vetores que contem os caminhos das imagens para...
    bowManager->pathsToImages_dictionary = vectorSamplesUsed2CreateDict;//criar o dicionario
    bowManager->pathsToImages_train = vectorSamplesUsed2TrainClassifier;//treinar o classificador
    bowManager->pathsToImages_test = vectorSamplesUsed2TestClassifier;//testar o classificador
    //////////////////////////////////////////////////////////////////////
    bowManager->imageSamplerFunction = gridSamplingBow;//ponteiro da funcao para o sampling
    ArgumentList* gridSamplingArguments = createArgumentList();
    ARGLIST_PUSH_BACK_AS(size_t,gridSamplingArguments,64); //patch size X
    ARGLIST_PUSH_BACK_AS(size_t,gridSamplingArguments,64); //patch size Y
    bowManager->argumentListOfSampler = gridSamplingArguments;//passando a lista de argumentos para o bow manager
    //////////////////////////////////////////////////////////
    bowManager->freeFunction2SamplerOutput = destroyImageVoidPointer;
    /////////////////////////////////////////////////////////////////
    bowManager->featureExtractorFunction = computeMergedFeats;//ponteiro da funcao para a extracao de features
    ArgumentList* mergedArguments = createArgumentList();
    size_t nbins = 4; // Bins de cor por canal
    ARGLIST_PUSH_BACK_AS(size_t,mergedArguments,nbins); //nBins per channel
    ARGLIST_PUSH_BACK_AS(size_t,mergedArguments,nbins*nbins*nbins); //total number of channels
    nbins = 9; // Bins angulares
    ARGLIST_PUSH_BACK_AS(size_t, mergedArguments, nbins);
    bowManager->argumentListOfFeatureExtractor = mergedArguments;
    ///////////////////////////////////////
    bowManager->distanceFunction = computeNormalizedL1Norm;
    bowManager->argumentListOfDistanceFunction = NULL;
    ////////////////////////////////////////////////////
    bowManager->clusteringFunction = kmeansClusteringBow;
    ArgumentList* clusteringMethodArguments = createArgumentList();
    ARGLIST_PUSH_BACK_AS(size_t,clusteringMethodArguments,numberOfVisualWords); //number of words
    ARGLIST_PUSH_BACK_AS(size_t,clusteringMethodArguments,100); //maximum number of iterations
    ARGLIST_PUSH_BACK_AS(double,clusteringMethodArguments, 1e-4); //tolerance
    ARGLIST_PUSH_BACK_AS(int,clusteringMethodArguments,0); //seed
    ARGLIST_PUSH_BACK_AS(DistanceFunction,clusteringMethodArguments,computeNormalizedL1Norm); //seed
    ARGLIST_PUSH_BACK_AS(ArgumentList*,clusteringMethodArguments,NULL); //seed
    bowManager->argumentListOfClustering = clusteringMethodArguments;
    ///////////////////////////////////////////////////////////////
    computeDictionary(bowManager);
    //////////////////////
    bowManager->mountHistogramFunction = computeCountHistogram_bow;
    bowManager->argumentListOfHistogramMounter = NULL;
    ///////////////////////
    SVM_Classifier* classifiersvm = createSVMClassifier();
    classifiersvm->param.kernel_type = RBF;
    classifiersvm->param.gamma = 3.5;
    bowManager->classifier = (void*)classifiersvm;
    bowManager->fitFunction = svm_Classifier_fit;
    bowManager->storeTrainData = false;
    bowManager->predictFunction = svm_Classifier_predict;
    bowManager->storePredictedData = false;
    bowManager->freeFunctionClassifier = destroySVMClassifierForVoidPointer;
    //////////////////////////////////////
    trainClassifier(bowManager);
    //////////
    GVector* labelsPredicted = predictLabels(bowManager);
    //////////////////////////

    //////////////////////////
    //Le os true labels das imagens e checa com os labels predizidos pelo o classificador.
    //computa uma simples acuracia (numero de amostras rotuladas corretamente / numero de amostras do conjunto)
    GVector* trueLabels = createNullVector(bowManager->pathsToImages_test->size,sizeof(int));
    int hit = 0;
    printf("file | predicted true\t\tcorrect\n");
    char symbol;
    for (size_t index = 0; index < bowManager->pathsToImages_test->size; ++index) {
        symbol = 'X';
        char * path = VECTOR_GET_ELEMENT_AS(char*,bowManager->pathsToImages_test,index);
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
    double acuracia = ((double)hit)/bowManager->pathsToImages_test->size;
    printf("acuracia: %f\n",acuracia);
    /////////////////////////////////////
    destroyBagOfVisualWordsManager(&bowManager);
    destroyVector(&trueLabels);
    destroyVector(&labelsPredicted);
    return 0;

}


