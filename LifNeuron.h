#pragma once
#include <vector>
#include <math.h>

class Synapse;

class LifNeuron {
public:
    void UpdatePotential(int time, float strength);

    LifNeuron( const std::vector<const Synapse &> &outputSynapses, float v, float x, float a, float vMinThresh,
               float vMaxThresh, float tau, float tps, float tOut, float tRef, bool isConsistent );

private:
    std::vector<const Synapse &> outputSynapses;
    float v;
    float x;
    float a;
    float vMinThresh;
    float vMaxThresh;
    float tau;
    float tps;
    float tOut;
    float tRef;
    bool isConsistent;

    float GetWDyn(float tout, float tp, float tRef);
};
