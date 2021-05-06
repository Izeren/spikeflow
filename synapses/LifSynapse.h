#pragma once

#include "ISynapse.h"

class LifSynapse : public ISynapse {

public:
    explicit LifSynapse(
            bool _isUpdatable,
            SPIKING_NN::Strength _strength,
            SPIKING_NN::Time _delay,
            INeuron *_prev,
            INeuron *_next );

    void SetUpdatable( bool updatable );

    float GetDlDw() const;

    void SetGrad( float DlDw );

    void RegisterPreSynapticSpike( SPIKING_NN::Time time ) override;

    void RegisterPostSynapticSpike( SPIKING_NN::Time time ) override;

    void GradStep( float learningRateV ) override;

    float GetGrad() const override;

    void ResetGrad() override;

    void Backward( float potential ) override;

protected:
    float DlDw;
    float batchDlDw;
};