#pragma once

#include <random>
#include "INeuron.h"

class BasicNeuron : public INeuron {
public:
    void ProcessInputSpike( SPIKING_NN::Time time, SPIKING_NN::Potential potential ) override;

    void Reset() override;

    void ResetGrad() override;

    void RandomInit( float alpha, size_t layerSize, size_t nextLayerSize ) override;

    BasicNeuron();

    void RelaxOutput( SPIKING_NN::Time time, bool withSpike ) override;

    void Backward( float layerTotalOutput, float delta ) override;

    void SetGrad( float _grad );

    float GetGrad() const override;

    void GradStep( float learningRate ) override;


    float GetOutput() const override;

    float GetMaxMP() override;

    void NormalizePotential( SPIKING_NN::Time time ) override;

    SPIKING_NN::Time GetFirstSpikeTS() override;

protected:
    float grad;
};
