#pragma once

#include "SpikingGeneral.h"


class INeuron;

class ISynapse {

public:

    const static SPIKING_NN::Time DEFAULT_SYNAPSE_DELAY;
    const static SPIKING_NN::Strength DEFAULT_SYNAPSE_STRENGTH;
    const static SPIKING_NN::Strength DEFAULT_LATERAL_SYNAPSE_STRENGTH;
    const static bool DEFAULT_SYNAPSE_UPDATABILITY;


    const static SPIKING_NN::Strength outputSpikeStrength;

    const static SPIKING_NN::Strength inputSpikeStrength;

    const static SPIKING_NN::Time tauOutput;

    const static SPIKING_NN::Time tauInput;

    ISynapse( bool _isUpdatable, SPIKING_NN::Strength _strength, SPIKING_NN::Time _delay );

    explicit ISynapse( bool _isUpdatable = DEFAULT_SYNAPSE_UPDATABILITY,
                       SPIKING_NN::Strength strength = DEFAULT_SYNAPSE_STRENGTH,
                       SPIKING_NN::Time delay = ISynapse::DEFAULT_SYNAPSE_DELAY,
                       INeuron *preSynapticNeuron = nullptr,
                       INeuron *postSynapticNeuron = nullptr );

    virtual SPIKING_NN::Time GetDelay();

    virtual SPIKING_NN::Strength GetStrength();

    virtual ~ISynapse();

    virtual void GradStep( float learningRateV, size_t activeNeurons, size_t nextLayerSize, float weightNormFactor ) = 0;

    virtual float GetGrad() const = 0;

    virtual void ResetGrad() = 0;

    virtual void Backward( float potential ) = 0;

    bool IsUpdatable();

    INeuron *GetPreSynapticNeuron() const;

    void SetPreSynapticNeuron( INeuron *preSynapticNeuron );

    INeuron *GetPostSynapticNeuron() const;

    void SetPostSynapticNeuron( INeuron *postSynapticNeuron );

    virtual void RegisterPreSynapticSpike( SPIKING_NN::Time time );

    virtual void RegisterPostSynapticSpike( SPIKING_NN::Time time );

    static SPIKING_NN::Strength GetPreSynapticUpdateStrength( SPIKING_NN::Strength strength );

    static SPIKING_NN::Strength GetPostSynapticUpdateStrength( SPIKING_NN::Strength strength );

    void SetStrength( SPIKING_NN::Strength strength );

    void SetDelay( SPIKING_NN::Time delay );

protected:
    bool updatable;
    SPIKING_NN::Strength strength;
    SPIKING_NN::Time delay;
    INeuron *preSynapticNeuron;
    INeuron *postSynapticNeuron;

    SPIKING_NN::Potential inputTrace;
    SPIKING_NN::Potential outputTrace;
    SPIKING_NN::Time inputRelaxation;
    SPIKING_NN::Time outputRelaxation;

};