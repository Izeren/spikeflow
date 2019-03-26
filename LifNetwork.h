//
// Created by izeren on 3/25/19.
//
#pragma once

#include <map>
#include "INetwork.h"

class LifNetwork : public INetwork {
public:
    explicit LifNetwork( size_t inputSize = 0, size_t outputSize = 0 );

    ~LifNetwork() override;

    void AddNeuron( size_t neuronId, SPIKING_NN::NEURON_TYPE neuronType ) override;

    void AddLink( size_t preSynapticNeuronId, size_t postSynapticNeuronId, SPIKING_NN::Strength strength,
                  SPIKING_NN::Time delay ) override;


protected:
    std::map<size_t, INeuron *> neuronMap;
    size_t size;

    void Reset() override;


};
