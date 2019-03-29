//
// Created by izeren on 3/25/19.
//
#pragma once

#include <map>
#include "INetwork.h"
#include "INeuron.h"


template<class Neuron, class Synapse, class EventManager>
class BasicNetwork : public INetwork {
public:
    explicit BasicNetwork( size_t inputSize = 0, size_t outputSize = 0, bool useSTDP = true );

    ~BasicNetwork() override;

    void AddNeuron( size_t neuronId, SPIKING_NN::NEURON_TYPE neuronType );

    void AddLink( size_t preSynapticNeuronId, size_t postSynapticNeuronId, SPIKING_NN::Strength strength,
                  SPIKING_NN::Time delay );

protected:
    size_t size;

    void Reset() override;

};

template<class Neuron, class Synapse, class EventManager>
BasicNetwork<Neuron, Synapse, EventManager>::BasicNetwork( size_t inputSize, size_t outputSize, bool useSTDP ):
        INetwork( useSTDP ) {
    size = 0;
    for ( auto neuronId = 0; neuronId < inputSize; ++neuronId ) {
        AddNeuron( size++, SPIKING_NN::NEURON_TYPE::INPUT );
    }
    for ( auto neuronId = 0; neuronId < outputSize; ++neuronId ) {
        AddNeuron( size++, SPIKING_NN::NEURON_TYPE::OUTPUT );
    }
    eventManager = new EventManager();
}

template<class Neuron, class Synapse, class EventManager>
BasicNetwork<Neuron, Synapse, EventManager>::~BasicNetwork() {
    for ( auto id_ptr: neuronMap ) {
        delete id_ptr.second;
    }
    delete eventManager;
}

template<class Neuron, class Synapse, class EventManager>
void BasicNetwork<Neuron, Synapse, EventManager>::AddNeuron( size_t neuronId, SPIKING_NN::NEURON_TYPE neuronType ) {
    INetwork::AddNeuron( new Neuron(), neuronId, neuronType );
}

template<class Neuron, class Synapse, class EventManager>
void BasicNetwork<Neuron, Synapse, EventManager>::AddLink( size_t preSynapticNeuronId, size_t postSynapticNeuronId,
                                                           SPIKING_NN::Strength strength,
                                                           SPIKING_NN::Time delay ) {
    INetwork::AddLink( new Synapse(), preSynapticNeuronId, postSynapticNeuronId, strength, delay );
}

template<class Neuron, class Synapse, class EventManager>
void BasicNetwork<Neuron, Synapse, EventManager>::Reset() {
    for ( auto id_ptr : neuronMap ) {
        id_ptr.second->Reset();
    }
}