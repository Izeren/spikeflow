#include "Synapse.h"


Synapse::Synapse( bool updatable, float strength, const Neuron &previous, const Neuron &next ) : updatable( updatable ),
                                                                                                 strength( strength ),
                                                                                                 previous( previous ),
                                                                                                 next( next ) {}
