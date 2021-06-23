#pragma once

#include "ISynapse.h"


class BasicSynapse : public ISynapse {
public:
    explicit BasicSynapse(
            bool _isUpdatable,
            SPIKING_NN::Strength _strength,
            SPIKING_NN::Time _delay,
            INeuron *_prev,
            INeuron *_next );

    void SetUpdatable( bool updatable );

    void RegisterPreSynapticSpike( SPIKING_NN::Time time ) override;

    void RegisterPostSynapticSpike( SPIKING_NN::Time time ) override;

    void GradStep( float learningRateV, size_t activeNeurons, size_t nextLayerSize, float weightNormFactor ) override;

    void Backward( float potential ) override;

    float GetGrad() const override;

    void ResetGrad() override;

    void SetGrad( float grad );

protected:
    float grad;
};