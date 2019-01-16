#include "LifNeuron.h"

float LifNeuron::GetWDyn(float tout, float tp, float tRef) {
    if (tout < 0 || tp < 0 || tRef < 0 || tp - tout < tRef) {
        return 1;
    } else {
        return (tp - tout) * (tp - tout) / tRef;
    }
}


void LifNeuron::UpdatePotential(int time, float strength) {
    if ( isConsistent ) {
        isConsistent = false;
    }
    float wDyn = GetWDyn(tOut, time, tRef);
    v = v * exp((tps - time) / tau) + strength * wDyn;
    tps = time;
}

LifNeuron::LifNeuron( const std::vector<const Synapse &> &outputSynapses, float v, float x, float a, float vMinThresh,
                      float vMaxThresh, float tau, float tps, float tOut, float tRef, bool isConsistent )
        : outputSynapses( outputSynapses ), v( v ), x( x ), a( a ), vMinThresh( vMinThresh ), vMaxThresh( vMaxThresh ),
          tau( tau ), tps( tps ), tOut( tOut ), tRef( tRef ), isConsistent( isConsistent ) {}
