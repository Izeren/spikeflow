#include "Synapse.h"
#include "INeuron.h"

Synapse::Synapse( bool _isUpdatable, float _strength, float _delay,
        INeuron *_previous, INeuron *_next ) :
        ISynapse(_strength, _delay, _previous, _next) {
    updatable = _isUpdatable;
    DaDx = 0;
    DlDw = 0;
}

float Synapse::getDaDx() const {
    return DaDx;
}

void Synapse::setDaDx( float DaDx ) {
    Synapse::DaDx = DaDx;
}

float Synapse::getDlDw() const {
    return DlDw;
}

void Synapse::setDlDw( float DlDw ) {
    Synapse::DlDw = DlDw;
}

bool Synapse::isUpdatable() const {
    return updatable;
}

void Synapse::setUpdatable( bool updatable ) {
    Synapse::updatable = updatable;
}
