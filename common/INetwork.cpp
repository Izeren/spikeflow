#include <LifSynapse.h>
#include "INetwork.h"
#include "INeuron.h"
#include "ISynapse.h"
#include "IEventManager.h"

void INetwork::Forward( const SPIKING_NN::Sample &sample, std::vector<float> &output, SPIKING_NN::Time time )
{
    eventManager->RegisterSample( sample, input );
    eventManager->RunSimulation( time, useSTDP );
    output.resize( this->output.size());
    for ( int neuronId = 0; neuronId < output.size(); ++neuronId ) {
        output[neuronId] = this->output[neuronId]->GetOutput();
    }
    Reset();
}

SPIKING_NN::Score INetwork::ScoreModel( SPIKING_NN::Dataset &data, SPIKING_NN::LossFunction lossFunction, bool onTest,
                                        SPIKING_NN::Time time )
{
    std::vector<SPIKING_NN::Sample> &samples = ( onTest ? data.xTest : data.xTrain );
    std::vector<SPIKING_NN::Target> &labels = ( onTest ? data.yTest : data.yTrain );
    std::vector<SPIKING_NN::Output> predictions( samples.size());
    for ( int sampleId = 0; sampleId < samples.size(); ++sampleId ) {
        Forward( samples[sampleId], predictions[sampleId], time );
    }
    return lossFunction( predictions, labels );
}

IEventManager *INetwork::GetEventManager() const
{
    return eventManager;
}

void INetwork::SetEventManager( IEventManager *eventManager )
{
    INetwork::eventManager = eventManager;
}

void INetwork::AddNeuron( INeuron *neuronPtr, size_t neuronId, SPIKING_NN::NEURON_TYPE neuronType )
{
    if ( neuronMap.find( neuronId ) == neuronMap.end()) {
        neuronMap[neuronId] = neuronPtr;
        if ( neuronType == SPIKING_NN::NEURON_TYPE::INPUT ) {
            input.push_back( neuronPtr );
        } else if ( neuronType == SPIKING_NN::NEURON_TYPE::OUTPUT ) {
            output.push_back( neuronPtr );
        }
    }
}

void INetwork::AddLink( size_t preSynapticNeuronId, size_t postSynapticNeuronId,
                        SPIKING_NN::Strength strength, SPIKING_NN::Time delay )
{
    INeuron *prev = neuronMap[preSynapticNeuronId];
    INeuron *next = neuronMap[postSynapticNeuronId];
    ISynapse *synapsePtr = new LifSynapse( true, strength, delay, prev, next );
    prev->AddOutputSynapse( synapsePtr );
    next->AddInputSynapse( synapsePtr );
}

INetwork::INetwork( bool useSTDP ) : useSTDP( useSTDP ) { }
