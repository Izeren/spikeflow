#pragma once

#include <unordered_set>

class ISynapse;


class INeuron {
public:
    virtual void ProcessInputSpike( float time, float potential ) = 0;

    INeuron( float potential = 0 );

    virtual ~INeuron();

    const std::unordered_set<ISynapse *> &GetInputSynapses() const;

    const std::unordered_set<ISynapse *> &GetOutputSynapses() const;

    void AddInputSynapse( ISynapse *synapse );

    void AddOutputSynapse( ISynapse *synapse );

    void ForgetInputSynapse( ISynapse *synapse );
    void ForgetOutputSynapse( ISynapse *synapse );




protected:
    std::unordered_set<ISynapse *> inputSynapses;
    std::unordered_set<ISynapse *> outputSynapses;
    float potential;
};