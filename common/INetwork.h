#pragma once

#include <vector>
#include "SpikingGeneral.h"

class INeuron;

class ISynapse;

class IEventManager;

class INetwork {
public:

    INetwork( bool useSTDP = true );

    virtual ~INetwork() = default;

    virtual void AddNeuron( INeuron *neuronPtr, size_t neuronId, SPIKING_NN::NEURON_TYPE neuronType );

    virtual void
    AddLink( size_t preSynapticNeuronId, size_t postSynapticNeuronId,
             SPIKING_NN::Strength strength, SPIKING_NN::Time delay );

    virtual void Forward( const SPIKING_NN::Sample &sample, std::vector<float> &output, SPIKING_NN::Time time );

    virtual SPIKING_NN::Score
    ScoreModel( SPIKING_NN::Dataset &data, SPIKING_NN::LossFunction function, bool onTest, SPIKING_NN::Time time );

    IEventManager *GetEventManager() const;

    void SetEventManager( IEventManager *eventManager );


protected:

    virtual void Reset() = 0;

    SPIKING_NN::Layer input;
    SPIKING_NN::Layer output;
    std::map<size_t, INeuron *> neuronMap;

    IEventManager *eventManager;
    bool useSTDP;

};
