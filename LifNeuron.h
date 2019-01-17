#pragma once
#include <vector>
#include <math.h>

class Synapse;

class LifNeuron {
public:
    void UpdatePotential(int time, float potential);

    LifNeuron( const std::vector<Synapse *> &outputSynapses, float v, float a, float vMinThresh,
               float vMaxThresh, float tau, float tps, float tOut, float tRef, bool isConsistent );
    const std::vector<Synapse *> &GetOutputSynapses() const;
    bool IsConsistent();
    bool NormalizePotential(int time);
    void AddSynapse(Synapse *synapse);

private:
    std::vector<Synapse *> outputSynapses;
    float v;
    float a;
    float vMinThresh;
    float vMaxThresh;
    float tau;
    float tps;
    float tOut;
    float tRef;
    bool isConsistent;

    float GetWDyn(float tOut, float tp, float tRef);
};
