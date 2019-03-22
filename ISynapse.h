#pragma once

class INeuron;

class ISynapse {

public:
    ISynapse( float _strength, float _delay );

    ISynapse( float strength = 1.0, float delay = 1.0, INeuron *preSynapticNeuron = 0, INeuron *postSynapticNeuron );

    virtual INeuron *GetPreSynapticNeuron() = 0;

    virtual INeuron *GetPostSynapticNeuron() = 0;

    virtual float GetDelay();

    virtual float GetStrength();

    virtual ~ISynapse();

protected:
    float strength;
    float delay;
    INeuron *preSynapticNeuron;
    INeuron *postSynapticNeuron;
};