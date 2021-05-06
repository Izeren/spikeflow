#pragma once

#include "ILayer.h"

class DenseVanillaLayer : public ILayer {
public:

    DenseVanillaLayer( std::string _name, size_t _size, const INeuronBuilder &_neuronBuilder );

    void Init( float alpha, size_t nextLayerSize ) override;

    ILayer &Relax( SPIKING_NN::Time time ) override;

    ILayer &LogBasicStats() override;

    ILayer &ResetStats() override;

    ILayer &ResetPotentials() override;

    ILayer &ResetGrad() override;

    ILayer &
    GradStep( size_t batchSize, float learningRateV, float learningRateW, float BETA, bool isInput ) override;

    ILayer &Backward( const std::vector<float> &deltas ) override;

    std::string ToString() const override;
};