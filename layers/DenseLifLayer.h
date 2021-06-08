#pragma once

#include "ILayer.h"

class DenseLifLayer : public ILayer {
public:

    explicit DenseLifLayer( LayerMeta meta );

    void Init( size_t nextLayerSize, std::default_random_engine &generator, float z ) override;

    ILayer &Relax( SPIKING_NN::Time time ) override;

    ILayer &LogBasicStats() override;

    ILayer &ResetStats() override;

    ILayer &ResetPotentials() override;

    ILayer &ResetGrad() override;

    ILayer &GradStep( size_t batchSize, float learningRateV, float learningRateW, float BETA, bool isInput ) override;

    ILayer &Backward( const std::vector<float> &deltas ) override;

    virtual ILayer &Forward() override;

    std::string ToString() const override;
};