#include "BasicSynapse.h"
#include "INeuron.h"

BasicSynapse::BasicSynapse(
        bool _isUpdatable,
        SPIKING_NN::Strength _strength,
        SPIKING_NN::Time _delay,
        INeuron *_prev,
        INeuron *_next ) : ISynapse( _isUpdatable, _strength, _delay, _prev, _next )
{
    grad = 0;
}

void BasicSynapse::SetUpdatable( bool updatable )
{
    BasicSynapse::updatable = updatable;
}

void BasicSynapse::RegisterPreSynapticSpike( SPIKING_NN::Time time ) { }

void BasicSynapse::RegisterPostSynapticSpike( SPIKING_NN::Time time ) { }

void BasicSynapse::GradStep( float learningRateV )
{
    strength -= grad * learningRateV;
}

float BasicSynapse::GetGrad() const
{
    return grad;
}

void BasicSynapse::SetGrad( float _grad )
{
    grad = _grad;
}

void BasicSynapse::Backward( float potential )
{
    grad = postSynapticNeuron->GetGrad() * potential;
}

void BasicSynapse::ResetGrad()
{
    grad = 0;
}
