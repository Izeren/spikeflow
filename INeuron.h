#pragma once

#include "SpikingGeneral.h"
#include <unordered_set>

class ISynapse;

typedef std::unordered_set<ISynapse *> ISynapses;


class INeuron {
public:
    virtual void ProcessInputSpike( SPIKING_NN::Time time, SPIKING_NN::Potential potential ) = 0;

    virtual void NormalizePotential( SPIKING_NN::Time time ) = 0;

    explicit INeuron( SPIKING_NN::Potential potential = 0, SPIKING_NN::Time tRef = SPIKING_NN::TIME_STEP,
                      bool isConsistent = true );

    virtual ~INeuron();

    virtual float GetOutput() = 0;

    virtual void Reset() = 0;

    const ISynapses &GetInputSynapses() const;

    const ISynapses &GetOutputSynapses() const;

    void AddInputSynapse( ISynapse *synapse );

    void AddOutputSynapse( ISynapse *synapse );

    void ForgetInputSynapse( ISynapse *synapse );

    void ForgetOutputSynapse( ISynapse *synapse );

    SPIKING_NN::Time GetTRef() const;

    void SetTRef( SPIKING_NN::Time tRef );

    bool IsConsistent() const;

    void SetConsistent( bool consistent );


protected:
    ISynapses inputSynapses;
    ISynapses outputSynapses;
    SPIKING_NN::Time potential;
    SPIKING_NN::Time tRef;
    bool consistent;
};