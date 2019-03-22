#pragma once

#include "SpikingGeneral.h"

class INeuron;

class ISynapse {

public:
    ISynapse( SPIKING_NN::Strength _strength, SPIKING_NN::Time _delay );

    explicit ISynapse( SPIKING_NN::Strength strength = 1.0, SPIKING_NN::Time delay = 1.0,
                       INeuron *preSynapticNeuron = nullptr,
                       INeuron *postSynapticNeuron = nullptr );

    virtual INeuron *GetPreSynapticNeuron() = 0;

    virtual INeuron *GetPostSynapticNeuron() = 0;

    virtual SPIKING_NN::Time GetDelay();

    virtual SPIKING_NN::Strength GetStrength();

    virtual ~ISynapse();

protected:
    SPIKING_NN::Strength strength;
    SPIKING_NN::Time delay;
    INeuron *preSynapticNeuron;
    INeuron *postSynapticNeuron;
};