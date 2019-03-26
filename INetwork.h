//
// Created by izeren on 3/22/19.
//
#pragma once

#include <vector>
#include "SpikingGeneral.h"

class INeuron;

class ISynapse;

class IEventManager;

class INetwork {
public:

    virtual ~INetwork() = default;

    virtual void AddNeuron( size_t neuronId, SPIKING_NN::NEURON_TYPE neuronType ) = 0;

    virtual void
    AddLink( size_t preSynapticNeuronId, size_t postSynapticNeuronId, SPIKING_NN::Strength strength,
             SPIKING_NN::Time delay ) = 0;

    virtual void Forward( const SPIKING_NN::Sample &sample, std::vector<float> &output, SPIKING_NN::Time time );

    virtual SPIKING_NN::Score
    ScoreModel( SPIKING_NN::Dataset &data, SPIKING_NN::LossFunction function, bool onTest, SPIKING_NN::Time time );

    IEventManager *GetEventManager() const;

    void SetEventManager( IEventManager *eventManager );


protected:

    virtual void Reset() = 0;

    SPIKING_NN::Layer input;
    SPIKING_NN::Layer output;

    IEventManager *eventManager;

};
