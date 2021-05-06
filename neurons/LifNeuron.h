#pragma once

#include <unordered_set>
#include <math.h>
#include "INeuron.h"

class ISynapse;


class LifNeuron : public INeuron {
public:
    void ProcessInputSpike( SPIKING_NN::Time time, SPIKING_NN::Potential potential ) override;

    void Reset() override;

    void ResetGrad() override;

    void RandomInit( float alpha, size_t layerSize, size_t nextLayerSize ) override;

    LifNeuron( SPIKING_NN::Potential v, SPIKING_NN::Potential a, SPIKING_NN::Potential vMinThresh,
               SPIKING_NN::Potential vMaxThresh, SPIKING_NN::Time tau, SPIKING_NN::Time tps, SPIKING_NN::Time tOut,
               SPIKING_NN::Time tRef, bool isConsistent );

    LifNeuron();

    float GetOutput() const override;

    float GetMaxMP() override;

    bool IsConsistent();

    void NormalizePotential( SPIKING_NN::Time time ) override;

    SPIKING_NN::Time GetFirstSpikeTS() override;

    void Backward( SPIKING_NN::Potential sumOutput, float delta ) override;

    void RelaxOutput( SPIKING_NN::Time time, bool withSpike ) override;

    void SetDlDV( float _gradV );

    void SetGrad( float _grad );

    float GetGrad() const override;

    float GetDlDv() const;

    SPIKING_NN::Time GetTau() const;

    float GetSigmaMu() const;

    void GradStep( float learningRate ) override;

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
