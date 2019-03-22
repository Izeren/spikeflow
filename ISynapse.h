#pragma once

#include "SpikingGeneral.h"

class INeuron;

class ISynapse {

public:
    ISynapse( SPIKING_NN::Strength _strength, SPIKING_NN::Time _delay );

    explicit ISynapse( SPIKING_NN::Strength strength = 1.0, SPIKING_NN::Time delay = 1.0,
                       INeuron *preSynapticNeuron = nullptr,
                       INeuron *postSynapticNeuron = nullptr );

    virtual SPIKING_NN::Time GetDelay();

    virtual SPIKING_NN::Strength GetStrength();

    virtual ~ISynapse();

    INeuron *GetPreSynapticNeuron() const;

    void SetPreSynapticNeuron( INeuron *preSynapticNeuron );

    INeuron *GetPostSynapticNeuron() const;

    void SetPostSynapticNeuron( INeuron *postSynapticNeuron );

protected:
    SPIKING_NN::Strength strength;
public:
    void SetStrength( SPIKING_NN::Strength strength );

    void SetDelay( SPIKING_NN::Time delay );

protected:
    SPIKING_NN::Time delay;
    INeuron *preSynapticNeuron;
    INeuron *postSynapticNeuron;

};