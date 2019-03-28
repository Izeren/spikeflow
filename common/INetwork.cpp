//
// Created by izeren on 3/22/19.
//

#include "INetwork.h"
#include "INeuron.h"
#include "IEventManager.h"

void INetwork::Forward( const SPIKING_NN::Sample &sample, std::vector<float> &output, SPIKING_NN::Time time ) {
    eventManager->RegisterSample( sample, input );
    eventManager->RunSimulation( time );
    output.resize( this->output.size());
    for ( int neuronId = 0; neuronId < output.size(); ++neuronId ) {
        output[neuronId] = this->output[neuronId]->GetOutput();
    }
    Reset();
}

SPIKING_NN::Score INetwork::ScoreModel( SPIKING_NN::Dataset &data, SPIKING_NN::LossFunction lossFunction, bool onTest,
                                        SPIKING_NN::Time time ) {
    std::vector<SPIKING_NN::Sample> &samples = (onTest ? data.xTest : data.xTrain);
    std::vector<SPIKING_NN::Target> &labels = (onTest ? data.yTest : data.yTrain);
    std::vector<SPIKING_NN::Output> predictions(samples.size());
    for ( int sampleId = 0; sampleId < samples.size(); ++sampleId ) {
        Forward( samples[sampleId], predictions[sampleId], time );
    }
    return lossFunction( predictions, labels );
}

IEventManager *INetwork::GetEventManager() const {
    return eventManager;
}

void INetwork::SetEventManager( IEventManager *eventManager ) {
    INetwork::eventManager = eventManager;
}
