#pragma once

#include "SpikingGeneral.h"
#include <unordered_set>
#include <random>

class ISynapse;

typedef std::unordered_set<ISynapse *> ISynapses;


class INeuron {
public:
    virtual void ProcessInputSpike( SPIKING_NN::Time time, SPIKING_NN::Potential potential ) = 0;

    virtual void NormalizePotential( SPIKING_NN::Time time ) = 0;

    explicit INeuron( SPIKING_NN::Potential potential = 0, SPIKING_NN::Time tRef = SPIKING_NN::TIME_STEP,
                      bool isConsistent = true );

    virtual ~INeuron();

    virtual float GetOutput() const = 0;

    virtual void Reset() = 0;

    virtual void RelaxOutput( SPIKING_NN::Time ts, bool withSpike ) = 0;

    virtual void Backward( float sumOutput, float delta ) = 0;

    virtual void GradStep( float learningRate ) = 0;

    virtual float GetGrad() const = 0;

    virtual void ResetGrad() = 0;

    virtual float GetMaxMP() = 0;

    virtual void RandomInit( float alpha, size_t layerSize, size_t nextLayerSize ) = 0;

    virtual SPIKING_NN::Time GetFirstSpikeTS() = 0;

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

    int GetInputSpikeCounter() const;

    int GetOutputSpikeCounter() const;

    bool operator<( const INeuron &other ) const;

protected:
    ISynapses inputSynapses;
    ISynapses outputSynapses;
    SPIKING_NN::Time potential;
    SPIKING_NN::Time tRef;
    bool consistent;
    int inputSpikeCounter;
    int outputSpikeCounter;
};