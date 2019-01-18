#pragma once
#include <vector>
#include <math.h>

class Synapse;

class LifNeuron {
public:
    void UpdatePotential(int time, float potential);

    LifNeuron( const std::vector<Synapse> &outputSynapses, float v, float a, float vMinThresh,
               float vMaxThresh, float tau, float tps, float tOut, float tRef, bool isConsistent );

    LifNeuron();

    const std::vector<Synapse> &GetOutputSynapses() const;
    bool IsConsistent();
    bool NormalizePotential(int time);
    void AddSynapse(const Synapse &synapse);
    void Backward(float sumA);

    std::vector<Synapse> outputSynapses;
    float v;
    float a;
    float vMinThresh;
    float vMaxThresh;
    float tau;
    float tps;
    float tOut;
    float tRef;
    bool isConsistent;
    float grad;
    float sigma_mu;
    float DlDV;

private:
    float GetWDyn(float tOut, float tp, float tRef);
};
