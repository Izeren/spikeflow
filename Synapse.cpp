#include "Synapse.h"
#include "LifNeuron.h"


Synapse::Synapse( bool updatable, float strength,
        LifNeuron *previous, LifNeuron *next ) :
        updatable( updatable ), strength( strength ),
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
}
