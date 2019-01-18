#include "Synapse.h"
#include "LifNeuron.h"


Synapse::Synapse( bool updatable, float strength, float x,
        LifNeuron *previous, LifNeuron *next ) :
        updatable( updatable ), strength( strength ), x ( x ),
        previous( previous ),
        next( next ) {
    DaDx = 0;
    DlDw = 0;
}

Synapse::Synapse() {
    previous = NULL;
    next = NULL;
    updatable = true;
    strength = 1;
    x = 0;
}
