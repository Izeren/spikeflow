#include "LifNeuron.h"
#include "Synapse.h"

float LifNeuron::GetWDyn(float tOut, float tp, float tRef) {
    if (tOut < 0 || tp < 0 || tRef < 0 || tp - tOut >= tRef) {
        return 1;
    } else {
        return (tp - tOut) * (tp - tOut) / tRef;
    }
}


void LifNeuron::UpdatePotential(int time, float potential) {
    if ( isConsistent ) {
        isConsistent = false;
    }
    float wDyn = GetWDyn(tOut, time, tRef);
    float leackyFactor = exp((tps - time) / tau);
    v = v * leackyFactor + potential * wDyn;
    tps = time;
}

LifNeuron::LifNeuron( const std::vector<Synapse *> &outputSynapses, float v, float a, float vMinThresh,
                      float vMaxThresh, float tau, float tps, float tOut, float tRef, bool isConsistent )
        : outputSynapses( outputSynapses ), v( v ), a( a ), vMinThresh( vMinThresh ), vMaxThresh( vMaxThresh ),
          tau( tau ), tps( tps ), tOut( tOut ), tRef( tRef ), isConsistent( isConsistent ) {}

const std::vector<Synapse *> &LifNeuron::GetOutputSynapses() const {
    return outputSynapses;
}

bool LifNeuron::IsConsistent() {
    return isConsistent;
}

bool LifNeuron::NormalizePotential(int time) {
    bool spikeStatus = false;
    if ( v < vMinThresh ) {
        v = vMinThresh;
    } else {
        if ( v >= vMaxThresh ) {
            spikeStatus = true;
            for ( auto synapse: outputSynapses ) {
                synapse->IncreaseX(exp((tOut - time - 1) / tau));
            }
            if ( tOut >= 0 ) {
                a += exp((tOut - time) / tau);
            } else {
                a += exp((-time) / tau);
            }
            tOut = time;
            while ( v >= vMaxThresh ) { v -= vMaxThresh; }
        }
    }
    isConsistent = true;
    return spikeStatus;
}

void LifNeuron::AddSynapse( Synapse *synapse ) {
    outputSynapses.push_back( synapse );
}
