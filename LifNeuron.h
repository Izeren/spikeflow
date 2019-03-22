#pragma once

#include <unordered_set>
#include <math.h>
#include "INeuron.h"

class ISynapse;

class LifNeuron : public INeuron {
public:
    void ProcessInputSpike( float time, float potential ) override;

    void Reset();

    LifNeuron( float v, float a, float vMinThresh, float vMaxThresh, float tau, float tps, float tOut,
               float tRef, bool isConsistent );

    LifNeuron();

    bool IsConsistent();

    bool NormalizePotential( float time );

    void Backward( float sumA );

    void RelaxOutput( float time, bool withSpike = false );

protected:

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
    float GetWDyn( float tOut, float tp, float tRef );
};
