#pragma once

#include "SpikingGeneral.h"

class INeuron;

class ISynapse {

public:

    static SPIKING_NN::Strength outputSpikeStrength;

    static SPIKING_NN::Strength inputSpikeStrength;

    static SPIKING_NN::Time tauOutput;

    static SPIKING_NN::Time tauInput;

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

    virtual void RegisterPreSynapticSpike( SPIKING_NN::Time time );

    virtual void RegisterPostSynapticSpike( SPIKING_NN::Time time );

    static SPIKING_NN::Strength GetPreSynapticUpdateStrength( SPIKING_NN::Strength strength );

    static SPIKING_NN::Strength GetPostSynapticUpdateStrength( SPIKING_NN::Strength strength );


protected:
    SPIKING_NN::Strength strength;
public:
    void SetStrength( SPIKING_NN::Strength strength );

    void SetDelay( SPIKING_NN::Time delay );

protected:
    SPIKING_NN::Time delay;
    INeuron *preSynapticNeuron;
    INeuron *postSynapticNeuron;

    SPIKING_NN::Potential inputTrace;
    SPIKING_NN::Potential outputTrace;
    SPIKING_NN::Time inputRelaxation;
    SPIKING_NN::Time outputRelaxation;

};