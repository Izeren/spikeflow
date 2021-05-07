#include <BasicNeuronBuilder.h>
#include <VanillaSynapseBuilder.h>
#include <IrisUtils.h>
#include <DataCommon.h>
#include <iostream>
#include <LifNeuronBuilder.h>
#include <LifSynapseBuilder.h>
#include <PreciseEventManager.h>
#include <LifNeuron.h>
#include <LifSynapse.h>
#include <BasicNetwork.hpp>
#include "DemoScripts.h"
#include "../../layers/DenseVanillaLayer.h"
#include "../../layers/DenseLifLayer.h"
#include "../../layers/ILayerBuilder.h"
#include "../../network/IDenseNetwork.h"

// TODO: Move to abstract losses
float SoftMaxLoss( const std::vector<float> &output, SPIKING_NN::Target classId,
                   std::vector<float> &softMax, std::vector<float> &deltas, float *S )
{
    float expSum = 0;
    float loss = 0;
    for ( auto activation: output ) {
        // TODO: make some exps on final target (important to understand everything around gradients formulas)
        softMax.emplace_back( exp( activation ));
        expSum += softMax.back();
    }
    for ( int neuronId = 0; neuronId < output.size(); neuronId++ ) {
        auto target = static_cast<float>( abs( classId - neuronId ) < SPIKING_NN::EPS );
        softMax[neuronId] /= expSum;
        deltas.emplace_back( softMax[neuronId] - target );
        *S += softMax[neuronId] * deltas[neuronId];
        loss += target * log( softMax[neuronId] );
    }
    return -loss;
}

// TODO: move to some utils
size_t GetClassPrediction( const std::vector<float> &output )
{
    size_t maxId = 0;
    float maxO = output[maxId];
    for ( size_t idx = 1; idx < output.size(); ++idx ) {
        if ( output[idx] > maxO ) {
            maxId = idx;
            maxO = output[idx];
        }
    }
    return maxO > 0 ? maxId : -1;
}

void DemoScripts::TrainVanillaIris( char *path, std::default_random_engine &generator )
{
    size_t IRIS_INPUT = 4;
    size_t IRIS_HIDDEN = 15;
    size_t IRIS_OUTPUT = 3;

    BasicNeuronBuilder neuronBuilder = BasicNeuronBuilder();

    DenseVanillaLayer input = DenseVanillaLayer( {0, "input", IRIS_INPUT, neuronBuilder} );
    DenseVanillaLayer hidden = DenseVanillaLayer( {0, "hidden1", IRIS_HIDDEN, neuronBuilder} );
    DenseVanillaLayer output = DenseVanillaLayer( {0, "output", IRIS_OUTPUT, neuronBuilder} );

    input.Init( hidden.GetSize());
    input.Init( output.GetSize());
    input.Init( 0 );

    VanillaSynapseBuilder synapseBuilder = VanillaSynapseBuilder( generator );
    input.BindWithNext( hidden, synapseBuilder );
    hidden.BindWithNext( output, synapseBuilder );

    SPIKING_NN::Dataset data;
    IRIS::ReadIris( path, data, 0.2 );
    DATA_CONVERSION::ConvertDataToTimings( data, 1 );

    float EPS = 1e-1;
    float LEARNING_RATE = 0.1;
    float SIMULATION_TIME = 1000;

    for ( int epochId = 0; epochId < SIMULATION_TIME; epochId++ ) {
        float totalLoss = 0.f;
        for ( int sampleId = 0; sampleId < data.xTrain.size(); sampleId++ ) {
            for ( int activationId = 0; activationId < input.GetSize(); ++activationId ) {
                input[activationId]->ProcessInputSpike( 0, data.xTrain[sampleId][activationId] );
            }
            hidden.Forward();
            output.Forward();

            std::vector<float> deltas;
            std::vector<float> softMax;
            float S = 0;
            auto result = std::vector<float>( output.GetSize());
            for ( size_t idx = 0; idx < output.GetSize(); ++idx ) {
                result[idx] = output[idx]->GetOutput();
            }
            float loss = SoftMaxLoss( result, data.yTrain[sampleId], softMax, deltas, &S );
            totalLoss += loss;

            output.Backward( deltas ), hidden.Backward( {} ), input.Backward( {} );

            output.GradStep( 1, LEARNING_RATE, LEARNING_RATE, 0, false );
            hidden.GradStep( 1, LEARNING_RATE, LEARNING_RATE, 0, false );
            input.GradStep( 1, LEARNING_RATE, LEARNING_RATE, 0, true );

            output.ResetPotentials().ResetGrad().ResetStats();
            hidden.ResetPotentials().ResetGrad().ResetStats();
            input.ResetPotentials().ResetGrad().ResetStats();
        }
        std::cout << "Loss for epoch: " << epochId << " is: " << totalLoss / data.xTrain.size() << "\n";
    }
}

void DemoScripts::TrainSpikingIris( char *path, std::default_random_engine &generator )
{
    float EPS = 1e-1;
    float LEARNING_RATE_V = 0.00000;
    float LEARNING_RATE_W = 0.00001;
    float SIMULATION_TIME = 100;

    size_t IRIS_INPUT = 4;
    size_t IRIS_HIDDEN = 5;
    size_t IRIS_HIDDEN_2 = 5;
    size_t IRIS_OUTPUT = 3;

    int BATCH_SIZE = 20;

    const float ALPHA = 0.05;
    const float BETA = 10;


    std::vector<ILayer *> layers;
    LifSynapseBuilder synapseBuilder = LifSynapseBuilder( generator );
    LifNeuronBuilder neuronBuilder = LifNeuronBuilder();
    DenseLifLayerBuilder layerBuilder = DenseLifLayerBuilder();
    PreciseEventManager eventManager = PreciseEventManager();


    std::vector<LayerMeta> layersMeta = {
            {5,   "input",   IRIS_INPUT,    neuronBuilder},
            {1,   "hidden1", IRIS_HIDDEN,   neuronBuilder},
            {1,   "hidden2", IRIS_HIDDEN_2, neuronBuilder},
            {0.1, "output",  IRIS_OUTPUT,   neuronBuilder}
    };

    auto network = IDenseNetworkBuilder().Build( layersMeta, synapseBuilder, layerBuilder, eventManager );

    SPIKING_NN::Dataset rawData;
    SPIKING_NN::SpikeTrainDataset data;
    IRIS::ReadIris( path, rawData, 0 );
    DATA_CONVERSION::ConvertDataToSpikeTrains( rawData, data, generator, SIMULATION_TIME );

    for ( int epochId = 0; epochId < 1000; epochId++ ) {
        float totalLoss = 0.f;
        int silentSamples = 0;
        int guesses = 0;
        int cetozaCnt = 0;
        int versiCnt = 0;
        int virginicaCnt = 0;
        float cetozaDelta = 0;
        float versiDelta = 0;
        float viriginicaDelta = 0;

        for ( int sampleId = 0; sampleId < data.xTrain.size(); sampleId++ ) {
            std::vector<float> output = network->Forward( data.xTrain[sampleId], SIMULATION_TIME, false );

            std::vector<float> deltas;
            std::vector<float> softMax;
            float S = 0;
            float loss = SoftMaxLoss( output, data.yTrain[sampleId], softMax, deltas, &S );
            totalLoss += loss;
            int predictedClassId = GetClassPrediction( output );
            silentSamples += predictedClassId == -1 ? 1 : 0;
            cetozaCnt += predictedClassId == 0 ? 1 : 0;
            versiCnt += predictedClassId == 1 ? 1 : 0;
            virginicaCnt += predictedClassId == 2 ? 1 : 0;
            int guess = abs((float) predictedClassId - data.yTrain[sampleId] ) < EPS;
            guesses += guess;
            cetozaDelta += abs( deltas[0] );
            versiDelta += abs( deltas[1] );
            viriginicaDelta += abs( deltas[2] );

            network->Backward( deltas );

            if (( sampleId + 1 ) % BATCH_SIZE == 0 ) {
                network->GradStep( BATCH_SIZE, LEARNING_RATE_V, LEARNING_RATE_W, BETA );
            }
            network->Reset();
        }
        std::cout << "Loss for epoch: " << std::setw( 3 ) << epochId << " is: ";
        std::cout << std::fixed << std::setprecision( 5 ) << std::setw( 9 ) << totalLoss / data.xTrain.size() << " ";
        std::cout << "Predictions: cetoza-" << cetozaCnt << " versi-" << versiCnt << " virginica-" << virginicaCnt
                  << " ";
        std::cout << "deltas: cetoza-" << cetozaDelta << " versi-" << versiDelta << " virginica-" << viriginicaDelta
                  << " ";
        std::cout << "Accuracy: " << (float) guesses / data.xTrain.size() << " ";
        std::cout << "Silent samples: " << std::setw( 3 ) << silentSamples << " | ";
        std::cout << network->GetStringStats();
        network->ResetStats();
    }
    delete network;
}

void DemoScripts::RunDummyModel()
{
    const int INPUT_SIZE = 1;
    const int OUTPUT_SIZE = 1;
    auto model = BasicNetwork<LifNeuron, LifSynapse, PreciseEventManager>( INPUT_SIZE, OUTPUT_SIZE, true );

    model.AddLink( 0, 1, 100, 0.1 );
    model.AddLink( 0, 2, 100, 0.15 );
    model.AddLink( 0, 3, 100, 0.05 );
    model.AddLink( 0, 4, 100, 0.2 );

    std::vector<float> output( OUTPUT_SIZE );

    model.Forward( {0.5, 100, 100, 100}, output, 4 );

    for ( auto activation: output ) {
        std::cout << activation << " ";
    }
    std::cout << "\n";
}
