#pragma once

#include <set>
#include <unordered_set>
#include <math.h>
#include "INeuron.h"

class ISynapse;


class LifNeuron : public INeuron {
public:
    float ProcessInputSpike( SPIKING_NN::Time time, SPIKING_NN::Potential potential ) override;

    void Reset() override;

    void ResetGrad() override;

    void RandomInit( float alpha, size_t layerSize, size_t nextLayerSize, float z, std::uniform_real_distribution<float> &dist,
                     std::default_random_engine &generator ) override;

    LifNeuron( SPIKING_NN::Potential v, SPIKING_NN::Potential a, SPIKING_NN::Potential vMinThresh,
               SPIKING_NN::Potential vMaxThresh, SPIKING_NN::Time tau, SPIKING_NN::Time tps,
               SPIKING_NN::Time tOut, SPIKING_NN::Time tRef, bool isConsistent );

    LifNeuron();

    float GetOutput() const override;

    float GetMaxMP() override;

    bool IsConsistent();

    float NormalizePotential( SPIKING_NN::Time time ) override;

    SPIKING_NN::Time GetFirstSpikeTS() override;

    void Backward( float sumOutput, float delta, size_t totalNeurons, size_t activeNeurons,
                   float meanReversedSquaredThresholds ) override;

    void RelaxOutput( SPIKING_NN::Time time, bool withSpike ) override;

    void SetDlDV( float _gradV );

    void SetGrad( float _grad );

    float GetGrad() const override;

    float GetDlDv() const;

    SPIKING_NN::Time GetTau() const;

    float GetSigmaMu() const;

    void GradStep( float learningRate, size_t neurons, size_t inputSynapses, size_t inputActiveSynapses ) override;

    void SetSigmaMu( float sigmaMu );

    void SetVMaxThresh( float _vMaxTrhesh );

protected:

    SPIKING_NN::Potential a;
    SPIKING_NN::Potential vMaxThresh;
    SPIKING_NN::Time tau;
    SPIKING_NN::Time tps;
    SPIKING_NN::Time tOut;
    SPIKING_NN::Time fts;

    float grad;
    float sigma_mu;
    float DlDV;
    float batchDlDV;

private:
    float GetWDyn( SPIKING_NN::Time tOut, SPIKING_NN::Time tp, SPIKING_NN::Time tRef );
};
