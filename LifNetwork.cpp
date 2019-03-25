//
// Created by izeren on 3/25/19.
//

#include "LifNetwork.h"
#include "LifNeuron.h"
#include "PreciseEventManager.h"
#include "Synapse.h"

LifNetwork::LifNetwork( size_t inputSize, size_t outputSize ) {
    size = 0;
    for ( auto neuronId = 0; neuronId < inputSize; ++neuronId ) {
        AddNeuron( size++, SPIKING_NN::NEURON_TYPE::INPUT );
    }
    for ( auto neuronId = 0; neuronId < outputSize; ++neuronId ) {
        AddNeuron( size++, SPIKING_NN::NEURON_TYPE::OUTPUT );
    }
    eventManager = new PreciseEventManager();
}

void LifNetwork::AddNeuron( size_t neuronId, SPIKING_NN::NEURON_TYPE neuronType ) {
    if ( neuronMap.find( neuronId ) == neuronMap.end()) {
        neuronMap[neuronId] = new LifNeuron();
        if ( neuronType == SPIKING_NN::NEURON_TYPE::INPUT ) {
            input.push_back( neuronMap[neuronId] );
        } else if ( neuronType == SPIKING_NN::NEURON_TYPE::OUTPUT ) {
            output.push_back( neuronMap[neuronId] );
        }
    }
}

void LifNetwork::AddLink( size_t preSynapticNeuronId, size_t postSynapticNeuronId, SPIKING_NN::Strength strength,
                          SPIKING_NN::Time delay ) {
    INeuron *prev = neuronMap[preSynapticNeuronId];
    INeuron *next = neuronMap[postSynapticNeuronId];
    ISynapse *synapsePtr = new Synapse();
    prev->AddOutputSynapse( synapsePtr );
    next->AddInputSynapse( synapsePtr );
    synapsePtr->SetStrength( strength );
    synapsePtr->SetDelay( delay );
    synapsePtr->SetPreSynapticNeuron( prev );
    synapsePtr->SetPostSynapticNeuron( next );
}

LifNetwork::~LifNetwork() {
    for ( auto id_ptr: neuronMap ) {
        delete id_ptr.second;
    }
    delete eventManager;
}