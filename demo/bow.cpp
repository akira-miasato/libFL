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

    //////////////////////////////////////////////////////////////////////
    //metodo de sampling que vai ser usado para criar os patchs. Se vc passar NULL aqui o estrutura
    // do bow vai criar um vetor de tamanho 1 onde o unico elemento desse vetor vai ser a imagem.
    bowManager->imageSamplerFunction = gridSamplingBow;//ponteiro da funcao para o sampling
    //Nesta demo o metodo de sampling  usado é o grid. Entao eu vou criar um argument list
    //para colocar os parametros do metodo de grinding que eu fiz.
    //Note que o cabecalho geral para a funcao de sammpling e
    //GVector* minhaFuncaoDeSampling(Image* image, BagOfVisualWordsManager* bagOfVisualWordsManager);
    ArgumentList* gridSamplingArguments = createArgumentList();
    ARGLIST_PUSH_BACK_AS(size_t,gridSamplingArguments,64); //patch size X
    ARGLIST_PUSH_BACK_AS(size_t,gridSamplingArguments,64); //patch size Y
    bowManager->argumentListOfSampler = gridSamplingArguments;//passando a lista de argumentos para o bow manager
    //////////////////////////////////////////////////////////

//     ////////////////////////////////////////////////////////////////////
//     //Random sampling
//     ////////////////////////////////////////////////////////////////////
//     bowManager->imageSamplerFunction = randomSamplingBow;//ponteiro da funcao para o sampling
//     ArgumentList* randomSamplingArguments = createArgumentList();
//     ARGLIST_PUSH_BACK_AS(size_t,randomSamplingArguments,64); //patch size X
//     ARGLIST_PUSH_BACK_AS(size_t,randomSamplingArguments,64); //patch size Y
//     ARGLIST_PUSH_BACK_AS(size_t,randomSamplingArguments,4); //number of sampled patches
//     bowManager->argumentListOfSampler = randomSamplingArguments;//passando a lista de argumentos para o bow manager
//     ////////////////////////////////////////////////////////

    /////////////////////////////////////////////////////////////////////////////
    //Essa função serve como um garbage collector para o metodo do sampling. Ao final de
    //cada iteracao, ela limpa da memoria os patchs gerados.
    //Se por acaso seu metodo de sampling nao gerar um vetor de imagens (GVector* de Image*),
    //voce pode passar NULL, porém fique consciente que vai ter um pouco de memory leak.
    //Ao final do programa seu sistema operional vai limpar toda a sujeira.
    bowManager->freeFunction2SamplerOutput = destroyImageVoidPointer;
    //bowManager->freeFunction2SamplerOutput = NULL;
    /////////////////////////////////////////////////


//     /////////////////////////////////////////////////////////////////
//     //Neste exemplo eu irei usar o descritor de cores aprendindo em aula.
//     bowManager->featureExtractorFunction = computeColorHistogramBow;//ponteiro da funcao para a extracao de features
//     //o meu metodo para fazer o histograma de cores recebe 2 parametros (exlcuindo vetor de entrada)
//     //0 - vetor com as imagens dos patchs (esse argumento n'ao conta)
//     //1 - numeros de bins por canal
//     //2 - numero total de bins (bins por canal * numero de canais). Portanto, eu vou
//     //criar uma argumentList e colocar dois parametros nela.
//     //Note que o cabecalho geral para a funcao do extrator e
//     //Matrix* MinhaFuncaoFeatureExtractor(GVector* outputSampler, BagOfVisualWordsManager* bagOfVisualWordsManager);
//     ArgumentList* colorFeatureExtractorArguments = createArgumentList();
//     size_t nbins = 4;
//     ARGLIST_PUSH_BACK_AS(size_t,colorFeatureExtractorArguments,nbins); //nBins per channel
//     ARGLIST_PUSH_BACK_AS(size_t,colorFeatureExtractorArguments,nbins*nbins*nbins); //total number of channels
//     bowManager->argumentListOfFeatureExtractor = colorFeatureExtractorArguments; //passando a lista de argumentos do feature extractor para o bow manager
//     ///////////////////////////////////////


//     /////////////////////////////////////////////////////////////////
//     // HOG
//     bowManager->featureExtractorFunction = computeHOGPerChannelBow;//ponteiro da funcao para a extracao de features
//     //0 - vetor com as imagens dos patchs (esse argumento n'ao conta)
//     //1 - numeros de bins angulares
//     ArgumentList* hogArguments = createArgumentList();
//     size_t nbins = 9;
//     ARGLIST_PUSH_BACK_AS(size_t, hogArguments, nbins);
//     bowManager->argumentListOfFeatureExtractor = hogArguments;
//     ///////////////////////////////////////

    /////////////////////////////////////////////////////////////////
    // All features
    bowManager->featureExtractorFunction = computeMergedFeats;//ponteiro da funcao para a extracao de features
    //0 - vetor com as imagens dos patchs (esse argumento n'ao conta)
    //1 - numeros de bins de cores (colorHist)
    //2 - numero total de bins de cores (colorHist)
    //3 - numero de bins angulares (HOG)
    ArgumentList* mergedArguments = createArgumentList();
    size_t nbins = 4; // Bins de cor por canal
    ARGLIST_PUSH_BACK_AS(size_t,mergedArguments,nbins); //nBins per channel
    ARGLIST_PUSH_BACK_AS(size_t,mergedArguments,nbins*nbins*nbins); //total number of channels
    nbins = 9; // Bins angulares
    ARGLIST_PUSH_BACK_AS(size_t, mergedArguments, nbins);
    bowManager->argumentListOfFeatureExtractor = mergedArguments;
    ///////////////////////////////////////

    ///////////////////////////////////////////////////////
    //Existem muitas maneiras de computar distancias entre pontos e vetores. A mais comum delas talvez
    //seja a distancia Euclidianda (norma l2). Neste exemplo eu vou usar a norma l1.
    //Quando vc implementar seu metodo de sampling, feature extraxtion, ou clustering, voce
    //pode usar essa funcao distancia.
    bowManager->distanceFunction = computeNormalizedL1Norm;
    bowManager->argumentListOfDistanceFunction = NULL;
    ////////////////////////////////////////////////////


    /////////////////////////////////////////////////////
    //Aqui precisamos definir qual funcao de clsutering vamos usar para encontrar as palavras
    //do dicionario. Eu optei de usar o kmeans clustering. Meu metodo do kmeans recebe 6 parametros,
    //desta forma eu preciso criar uma ArgumentList com 6 parametros.
    //Note que o cabecalho geral para a funcao de clustering e
    //typedef Matrix* minhaFuncaoDeClustering(Matrix* outputFeatureExtractor_allSamples, BagOfVisualWordsManager* bagOfVisualWordsManager);
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

    ////////////
    //computa o dicionario
    computeDictionary(bowManager);
    /////////////

    //////////////////////
    //define a funcao para montar o histograma
    bowManager->mountHistogramFunction = computeCountHistogram_bow;
    bowManager->argumentListOfHistogramMounter = NULL;
    ///////////////////////


    /////////////////////////////////////////////////
    //criar um classificador e define os parametros
    //do classficiador. Em seguida, o bow manager recebe
    //o ponteiro do classificador. Desta forma o classificador
    //podera ser usado internamente dentro do bow manager.

    //knn
//    Knn_Classifier* classifierknn = createKnnClassifier();
//    classifierknn->k = 1;
//    classifierknn->nlabels = 100;
//    bowManager->classifier = (void*)classifierknn;
//    bowManager->fitFunction = knn_Classifier_fit;
//    bowManager->storeTrainData = false;
//    bowManager->predictFunction = knn_Classifier_predict;
//    bowManager->storePredictedData = false;
//    bowManager->freeFunctionClassifier = destroyKnnClassifierForVoidPointer;


    //"kmeans classifier"
//    Kmeans_Classifier* classifierkmeans = createKmeansClassifier();
//    classifierkmeans->nlabels = 100;
//    bowManager->classifier = (void*)classifierkmeans;
//    bowManager->fitFunction = kmeans_Classifier_fit;
//    bowManager->storeTrainData = false;
//    bowManager->predictFunction = kmeans_Classifier_predict;
//    bowManager->storePredictedData = false;
//    bowManager->freeFunctionClassifier = destroyKmeansClassifierForVoidPointer;


    //SVM Classifier
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

    ///////
    //monta os histogramas, le os label e em seguida treina o classificador
    std::vector<BagOfVisualWordsManager*> bowManagerVec;
    bowManagerVec.push_back(bowManager);
    trainClassifier(bowManagerVec);
    //////////

    /////////////////////////////////////////////////////
    //monta os histogramas e usa o classificador treinado para
    //classificar as amostras do conjunto de teste
    GVector* labelsPredicted = predictLabels(bowManagerVec);
    //////////////////////////

    //////////////////////////
    //Le os true labels das imagens e checa com os labels predizidos pelo o classificador.
    //computa uma simples acuracia (numero de amostras rotuladas corretamente / numero de amostras do conjunto)
    GVector* trueLabels = createNullVector(bowManagerVec[0]->pathsToImages_test->size,sizeof(int));
    int hit = 0;
    printf("file | predicted true\t\tcorrect\n");
    char symbol;
    for (size_t index = 0; index < bowManagerVec[0]->pathsToImages_test->size; ++index) {
        symbol = 'X';
        char * path = VECTOR_GET_ELEMENT_AS(char*,bowManagerVec[0]->pathsToImages_test,index);
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
    double acuracia = ((double)hit)/bowManagerVec[0]->pathsToImages_test->size;
    printf("acuracia: %f\n",acuracia);
    /////////////////////////////////////
//
    for(BagOfVisualWordsManager* manager : bowManagerVec){
        destroyBagOfVisualWordsManager(&manager);
    }
    destroyVector(&trueLabels);
    destroyVector(&labelsPredicted);
    return 0;

}


