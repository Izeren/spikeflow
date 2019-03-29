#include "Synapse.h"
#include "INeuron.h"
#include <iostream>

Synapse::Synapse( bool _isUpdatable, float _strength, float _delay,
                  INeuron *_previous, INeuron *_next ) :
        ISynapse( _strength, _delay, _previous, _next ) {
    updatable = _isUpdatable;
    DaDx = 0;
    DlDw = 0;
}

bool Synapse::IsUpdatable() const {
    return updatable;
}

void Synapse::SetUpdatable( bool updatable ) {
    Synapse::updatable = updatable;
}

float Synapse::GetDaDx() const {
    return DaDx;
}

void Synapse::SetDaDx( float DaDx ) {
    Synapse::DaDx = DaDx;
}

float Synapse::GetDlDw() const {
    return DlDw;
}

void Synapse::SetDlDw( float DlDw ) {
    Synapse::DlDw = DlDw;
}

void Synapse::RegisterPreSynapticSpike( SPIKING_NN::Time time ) {
    std::cout << "Registered presynaptic at time: " << time << "\n";
    ISynapse::RegisterPreSynapticSpike( time );
}

void Synapse::RegisterPostSynapticSpike( SPIKING_NN::Time time ) {
    std::cout << "Registered postsynaptic at time: " << time << "\n";
    ISynapse::RegisterPostSynapticSpike( time );
}
