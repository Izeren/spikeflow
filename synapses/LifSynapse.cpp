#include "LifSynapse.h"
#include "INeuron.h"
#include <iostream>

LifSynapse::LifSynapse(
        bool _isUpdatable,
        SPIKING_NN::Strength _strength,
        SPIKING_NN::Time _delay,
        INeuron *_prev,
        INeuron *_next ) : ISynapse( _isUpdatable, _strength, _delay, _prev, _next )
{
    DlDw = 0;
    batchDlDw = 0;
}

void LifSynapse::SetUpdatable( bool updatable )
{
    LifSynapse::updatable = updatable;
}

float LifSynapse::GetDlDw() const
{
    return DlDw;
}

void LifSynapse::SetGrad( float DlDw )
{
    LifSynapse::DlDw = DlDw;
}

void LifSynapse::RegisterPreSynapticSpike( SPIKING_NN::Time time )
{
    std::cout << "Registered presynaptic at time: " << time << "\n";
    ISynapse::RegisterPreSynapticSpike( time );
}

void LifSynapse::RegisterPostSynapticSpike( SPIKING_NN::Time time )
{
    std::cout << "Registered postsynaptic at time: " << time << "\n";
    ISynapse::RegisterPostSynapticSpike( time );
}

void LifSynapse::GradStep( float learningRateV )
{
    strength -= batchDlDw * learningRateV; // * sqrt(N / m);
//    synapses[synapseId].strength -= BETA * LAMBDA * synapses[synapseId].strength * F;
}

void LifSynapse::Backward( float potential )
{
    DlDw = postSynapticNeuron->GetGrad() * potential;
    batchDlDw += DlDw;
}

float LifSynapse::GetGrad() const
{
    return DlDw;
}

void LifSynapse::ResetGrad()
{
    DlDw = 0;
    batchDlDw = 0;
}
