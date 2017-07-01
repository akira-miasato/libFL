#include "FL.h"
#include <vector>
#include <iostream>

void usage(char **argv){
    std::cout << "Usage: " << argv[0] << "<dataset> <feat_type> <dict_size> <sampling_function> <patch_size> [<num_patches>]" << std::endl;
    std::cout << "       " << "dataset: \"coil100\" or \"mnist\" or \"cifar10\" or \"pubfig\"" << std::endl;
    std::cout << "       " << "feat_type: \"color_hist\" or \"hog\" or \"merged\"" << std::endl;
    std::cout << "       " << "dict_size: int" << std::endl;
    std::cout << "       " << "sampling_function: \"grid\" or \"random\" or \"ift\"" << std::endl;
    std::cout << "       " << "patch_size: int" << std::endl;
    std::cout << "       " << "num_patches: int (only used and mandatory in random sampling)" << std::endl;
}


int main(int argc, char **argv) {
    
    if(argc != 6 && argc != 7 ){
        usage(argv);
        return -1;
    }
    std::string sampling_function(argv[4]);
    if(argc != 7 && sampling_function == "random"){
        usage(argv);
        return -1;
    }
    std::string dataset(argv[1]);
    std::string featType(argv[2]);
    size_t numberOfVisualWords = atoi(argv[3]);
    size_t patch = atoi(argv[5]);

    // Datasets
    std::string dictList, trainList, devList;   
    if(dataset == "coil100"){
        dictList = "/home/akira-miasato/data/img/coil100_train.txt";
        trainList = "/home/akira-miasato/data/img/coil100_train.txt";
        devList = "/home/akira-miasato/data/img/coil100_dev.txt";
    }
    else if(dataset == "pubfig"){
        dictList = "/home/akira-miasato/data/img/pubfig_train.txt";
        trainList = "/home/akira-miasato/data/img/pubfig_train.txt";
        devList = "/home/akira-miasato/data/img/pubfig_dev.txt";
    }
    else if(dataset == "mnist"){
        dictList = "/home/akira-miasato/data/img/mnist_train.txt";
        trainList = "/home/akira-miasato/data/img/mnist_train.txt";
        devList= "/home/akira-miasato/data/img/mnist_dev.txt";
    }
    else if(dataset == "corel"){
        dictList = "/home/akira-miasato/data/img/corel_train.txt";
        trainList = "/home/akira-miasato/data/img/corel_train.txt";
        devList= "/home/akira-miasato/data/img/corel_dev.txt";
    }
    else{
        std::cout << "Invalid dataset \"" << dataset << "\"" << std::endl;
        return -1;
    }
    
    //cada posicao do vetor tem uma string para o caminho de uma imagem
    GVector* vectorSamplesUsed2CreateDict =  splitsLinesInTextFile(dictList.c_str());
    GVector* vectorSamplesUsed2TrainClassifier =  splitsLinesInTextFile(trainList.c_str());
    GVector* vectorSamplesUsed2TestClassifier =  splitsLinesInTextFile(devList.c_str());

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
    if(sampling_function == "grid") bowManager->imageSamplerFunction = gridSamplingBow;
    else if (sampling_function == "random") bowManager->imageSamplerFunction = randomSamplingBow;
    else if (sampling_function == "ift") bowManager->imageSamplerFunction = iftSamplingBow;
    else std::cout << "Invalid sampling_function \"" << sampling_function << "\"" << std::endl;
    ArgumentList* gridSamplingArguments = createArgumentList();
    ARGLIST_PUSH_BACK_AS(size_t,gridSamplingArguments,patch); //patch size X
    ARGLIST_PUSH_BACK_AS(size_t,gridSamplingArguments,patch); //patch size Y
    if (sampling_function == "random") ARGLIST_PUSH_BACK_AS(size_t,gridSamplingArguments,atoi(argv[6])); //patch size Y
    bowManager->argumentListOfSampler = gridSamplingArguments;//passando a lista de argumentos para o bow manager
    //////////////////////////////////////////////////////////
    bowManager->freeFunction2SamplerOutput = destroyImageVoidPointer;
    /////////////////////////////////////////////////////////////////
    size_t nbins_color = 4; // Bins de cor por canal
    size_t nbins_angle = 18; // Bins angulares
    if(featType == "color_hist"){
        bowManager->featureExtractorFunction = computeColorHistogramBow;//ponteiro da funcao para a extracao de features
        ArgumentList* args = createArgumentList();
        ARGLIST_PUSH_BACK_AS(size_t,args, nbins_color); //nBins per channel
        ARGLIST_PUSH_BACK_AS(size_t,args, nbins_color*nbins_color*nbins_color); //total number of channels
        bowManager->argumentListOfFeatureExtractor = args;        
    }
    else if (featType == "hog"){
        bowManager->featureExtractorFunction = computeHOGBow;//ponteiro da funcao para a extracao de features
        ArgumentList* args = createArgumentList();
        ARGLIST_PUSH_BACK_AS(size_t, args, nbins_angle);
        bowManager->argumentListOfFeatureExtractor = args;        
    }
    else if (featType == "merged"){
        bowManager->featureExtractorFunction = computeMergedFeats;//ponteiro da funcao para a extracao de features
        ArgumentList* args = createArgumentList();
        ARGLIST_PUSH_BACK_AS(size_t,args, nbins_color); //nBins per channel
        ARGLIST_PUSH_BACK_AS(size_t,args, nbins_color*nbins_color*nbins_color); //total number of channels
        ARGLIST_PUSH_BACK_AS(size_t, args, nbins_angle);
        bowManager->argumentListOfFeatureExtractor = args;
    }
    else {
        std::cout << "Invalid feat_type \"" << featType << "\"" << std::endl;
        return -1;
    }

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


