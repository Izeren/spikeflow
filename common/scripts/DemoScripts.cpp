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

void ForwardVanillaLayer( ILayer &layer )
{
    std::vector<float> output = std::vector<float>( layer.size );
    for ( auto neuron: layer.neurons ) {
        float inPotential = 0;
        for ( auto synapse: neuron->GetInputSynapses()) {
            inPotential += synapse->GetPreSynapticNeuron()->GetOutput() * synapse->GetStrength();
        }
        neuron->ProcessInputSpike( 0, inPotential );
    }
}

// TODO: Move to abstract losses
float SoftMaxLoss( ILayer &outputLayer, SPIKING_NN::Target classId,
                   std::vector<float> &softMax, std::vector<float> &deltas, float *S )
{
    float expSum = 0;
    float loss = 0;
    for ( auto neuron: outputLayer.neurons ) {
        // TODO: make some exps on final target (important to understand everything around gradients formulas)
        softMax.emplace_back( exp( neuron->GetOutput()));
        expSum += softMax.back();
    }
    for ( int neuronId = 0; neuronId < outputLayer.size; neuronId++ ) {
        auto target = static_cast<float>( abs( classId - neuronId ) < SPIKING_NN::EPS );
        softMax[neuronId] /= expSum;
        deltas.emplace_back( softMax[neuronId] - target );
        *S += softMax[neuronId] * deltas[neuronId];
        loss += target * log( softMax[neuronId] );
    }
    return -loss;
}

// TODO: move to some utils
int GetClassPrediction( const ILayer &output )
{
    int maxId = 0;
    float maxO = output[maxId]->GetOutput();
    for ( int idx = 1; idx < output.size; ++idx ) {
        if ( output[idx]->GetOutput() > maxO ) {
            maxId = idx;
            maxO = output[idx]->GetOutput();
        }
    }
    return maxO > 0 ? maxId : -1;
}

void DemoScripts::TrainVanillaIris( char *path, std::default_random_engine &generator )
{
    int IRIS_INPUT = 4;
    int IRIS_HIDDEN = 15;
    int IRIS_OUTPUT = 3;

    BasicNeuronBuilder neuronBuilder = BasicNeuronBuilder();

    DenseVanillaLayer input = DenseVanillaLayer( "input", IRIS_INPUT, neuronBuilder );
    DenseVanillaLayer hidden = DenseVanillaLayer( "hidden1", IRIS_HIDDEN, neuronBuilder );
    DenseVanillaLayer output = DenseVanillaLayer( "output", IRIS_OUTPUT, neuronBuilder );

    input.Init( 3, hidden.size );
    input.Init( 3, output.size );
    input.Init( 3, 0 );

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
            for ( int activationId = 0; activationId < input.size; ++activationId ) {
                input[activationId]->ProcessInputSpike( 0, data.xTrain[sampleId][activationId] );
            }
            ForwardVanillaLayer( hidden );
            ForwardVanillaLayer( output );

            std::vector<float> deltas;
            std::vector<float> softMax;
            float S = 0;
            float loss = SoftMaxLoss( output, data.yTrain[sampleId], softMax, deltas, &S );
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

    int IRIS_INPUT = 4;
    int IRIS_HIDDEN = 5;
    int IRIS_HIDDEN_2 = 5;
    int IRIS_OUTPUT = 3;

    int BATCH_SIZE = 20;

    const float ALPHA = 0.05;
    const float BETA = 10;

    LifNeuronBuilder neuronBuilder = LifNeuronBuilder();

    DenseLifLayer input = DenseLifLayer( "input", IRIS_INPUT, neuronBuilder );
    DenseLifLayer hidden1 = DenseLifLayer( "hidden1", IRIS_HIDDEN, neuronBuilder );
    DenseLifLayer hidden2 = DenseLifLayer( "hidden2", IRIS_HIDDEN_2, neuronBuilder );
    DenseLifLayer output = DenseLifLayer( "output", IRIS_OUTPUT, neuronBuilder );

    input.Init( 5, hidden1.size );
    hidden1.Init( 1, hidden2.size );
    hidden2.Init( 1, output.size );
    output.Init( 0.1, 0 );

    LifSynapseBuilder synapseBuilder = LifSynapseBuilder( generator );
    input.BindWithNext( hidden1, synapseBuilder )
            .BindWithNext( hidden2, synapseBuilder )
            .BindWithNext( output, synapseBuilder );

    SPIKING_NN::Dataset rawData;
    SPIKING_NN::SpikeTrainDataset data;
    IRIS::ReadIris( path, rawData, 0 );
    DATA_CONVERSION::ConvertDataToUniformSpikeTrains( rawData, data, SIMULATION_TIME );

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
            PreciseEventManager eventManager = PreciseEventManager();
            eventManager.RegisterSpikeTrain( data.xTrain[sampleId], input );
            eventManager.RunSimulation( SIMULATION_TIME, false );

//            Order doesn't matter (output relaxation depends only on total simulation time)
            input.Relax( SIMULATION_TIME ).LogBasicStats();
            hidden1.Relax( SIMULATION_TIME ).LogBasicStats();
            hidden2.Relax( SIMULATION_TIME ).LogBasicStats();
            output.Relax( SIMULATION_TIME ).LogBasicStats();

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

            output.Backward( deltas ), hidden2.Backward( {} ), hidden1.Backward( {} ), input.Backward( {} );

            if (( sampleId + 1 ) % BATCH_SIZE == 0 ) {
                output.GradStep( BATCH_SIZE, LEARNING_RATE_V, LEARNING_RATE_W, BETA, false );
                hidden2.GradStep( BATCH_SIZE, LEARNING_RATE_V, LEARNING_RATE_W, BETA, false );
                hidden1.GradStep( BATCH_SIZE, LEARNING_RATE_V, LEARNING_RATE_W, BETA, false );
                input.GradStep( BATCH_SIZE, LEARNING_RATE_V, LEARNING_RATE_W, BETA, true );
                // It's important not to reset grad before all grad are performed
                output.ResetGrad();
                hidden2.ResetGrad();
                hidden1.ResetGrad();
                input.ResetGrad();
            }
            output.ResetPotentials(), hidden2.ResetPotentials(), hidden1.ResetPotentials(), input.ResetPotentials();
        }
        std::cout << "Loss for epoch: " << std::setw( 3 ) << epochId << " is: ";
        std::cout << std::fixed << std::setprecision( 5 ) << std::setw( 9 ) << totalLoss / data.xTrain.size() << " ";
        std::cout << "Predictions: cetoza-" << cetozaCnt << " versi-" << versiCnt << " virginica-" << virginicaCnt
                  << " ";
        std::cout << "deltas: cetoza-" << cetozaDelta << " versi-" << versiDelta << " virginica-" << viriginicaDelta
                  << " ";
        std::cout << "Accuracy: " << (float) guesses / data.xTrain.size() << " ";
        std::cout << "Silent samples: " << std::setw( 3 ) << silentSamples << " | ";
        std::cout << input << "\n";
        std::cout << hidden1 << "\n";
        std::cout << hidden2 << "\n";
        std::cout << output << "\n";
        input.ResetStats(), hidden1.ResetStats(), hidden2.ResetStats(), output.ResetStats();
    }
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
