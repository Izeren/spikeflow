//
// Created by izeren on 3/25/19.
//
#pragma once

#include <map>
#include "INetwork.h"


template<class Neuron, class Synapse, class EventManager>
class BasicNetwork : public INetwork {
public:
    explicit BasicNetwork( size_t inputSize = 0, size_t outputSize = 0 ) {
        size = 0;
        for ( auto neuronId = 0; neuronId < inputSize; ++neuronId ) {
            AddNeuron( size++, SPIKING_NN::NEURON_TYPE::INPUT );
        }
        for ( auto neuronId = 0; neuronId < outputSize; ++neuronId ) {
            AddNeuron( size++, SPIKING_NN::NEURON_TYPE::OUTPUT );
        }
        eventManager = new EventManager();
    }

    ~BasicNetwork() override {
        for ( auto id_ptr: neuronMap ) {
            delete id_ptr.second;
        }
        delete eventManager;
    }

    void AddNeuron( size_t neuronId, SPIKING_NN::NEURON_TYPE neuronType ) {
        INetwork::AddNeuron( new Neuron(), neuronId, neuronType );
    }

    void AddLink( size_t preSynapticNeuronId, size_t postSynapticNeuronId, SPIKING_NN::Strength strength,
                  SPIKING_NN::Time delay ) {
        INetwork::AddLink( new Synapse(), preSynapticNeuronId, postSynapticNeuronId, strength, delay );
    }


protected:
    size_t size;

    void Reset() override {
        for ( auto id_ptr: neuronMap ) {
            id_ptr.second->Reset();
        }
    }

};
