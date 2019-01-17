#include "Synapse.h"
#include "LifNeuron.h"


Synapse::Synapse( bool updatable, float strength, float x,
        LifNeuron &previous, LifNeuron &next ) :
        updatable( updatable ), strength( strength ), x ( x ),
        previous( previous ),
        next( next ) {}

LifNeuron &Synapse::GetNext() const {
    return next;
}

float Synapse::GetStrength() const {
    return strength;
}

void Synapse::IncreaseX( float delta ) {
    x += delta;
}
