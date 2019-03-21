#pragma once
#include <vector>
#include <math.h>
#include "INeuron.h"

class Synapse;

class LifNeuron : INeuron {
public:
    void UpdatePotential(int time, float potential);
    void Reset();

    LifNeuron( const std::vector<Synapse> &outputSynapses, float v, float a, float vMinThresh,
               float vMaxThresh, float tau, float tps, float tOut, float tRef, bool isConsistent );

    LifNeuron();

    const std::vector<Synapse> &GetOutputSynapses() const;
    bool IsConsistent();
    bool NormalizePotential(int time);
    void AddSynapse(const Synapse &synapse);
    void Backward(float sumA);
    void RelaxOutput( int time, bool withSpike=false );

    std::vector<Synapse> outputSynapses;
    float v;
    float a;
    float vMaxThresh;
    float tau;
    float tps;
    float tOut;
    float tRef;
    bool isConsistent;
    float grad;
    float sigma_mu;
    float DlDV;
    int spikeCounter;

private:
    float GetWDyn(float tOut, float tp, float tRef);
};
