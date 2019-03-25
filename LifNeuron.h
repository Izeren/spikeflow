#pragma once

#include <unordered_set>
#include <math.h>
#include "INeuron.h"

class ISynapse;


class LifNeuron : public INeuron {
public:
    void ProcessInputSpike( SPIKING_NN::Time time, SPIKING_NN::Potential potential ) override;

    void Reset();

    LifNeuron( SPIKING_NN::Potential v, SPIKING_NN::Potential a, SPIKING_NN::Potential vMinThresh,
               SPIKING_NN::Potential vMaxThresh, SPIKING_NN::Time tau, SPIKING_NN::Time tps, SPIKING_NN::Time tOut,
               SPIKING_NN::Time tRef, bool isConsistent );

    LifNeuron();

    float GetOutput() override;

    bool IsConsistent();

    void NormalizePotential( SPIKING_NN::Time time ) override;

    void Backward( SPIKING_NN::Potential sumA );

    void RelaxOutput( SPIKING_NN::Time time, bool withSpike = false );

protected:

    SPIKING_NN::Potential a;
    int spikeCounter;
    SPIKING_NN::Potential vMaxThresh;
    SPIKING_NN::Time tau;
    SPIKING_NN::Time tps;
    SPIKING_NN::Time tOut;
    SPIKING_NN::Time fts;

    float grad;
    float sigma_mu;
    float DlDV;

private:
    float GetWDyn( SPIKING_NN::Time tOut, SPIKING_NN::Time tp, SPIKING_NN::Time tRef );
};
