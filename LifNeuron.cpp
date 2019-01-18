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
    float leackyFactor;
    if ( tps < 0 ) {
        leackyFactor = 1;
    } else {
        leackyFactor = exp((tps - time) / tau);
    }

    v = v * leackyFactor + potential * wDyn;
    tps = time;
    spikeCounter += 1;
}

LifNeuron::LifNeuron( const std::vector<Synapse> &outputSynapses, float v, float a, float vMinThresh,
                      float vMaxThresh, float tau, float tps, float tOut, float tRef, bool isConsistent )
        : outputSynapses( outputSynapses ), v( v ), a( a ), vMaxThresh( vMaxThresh ),
          tau( tau ), tps( tps ), tOut( tOut ), tRef( tRef ), isConsistent( isConsistent ) {
    grad=0;
    sigma_mu=0.25;
    DlDV = 0;
}

const std::vector<Synapse> &LifNeuron::GetOutputSynapses() const {
    return outputSynapses;
}

bool LifNeuron::IsConsistent() {
    return isConsistent;
}

bool LifNeuron::NormalizePotential(int time) {
    bool spikeStatus = false;
    if ( v < -vMaxThresh) {
        v = -vMaxThresh;
    } else {
        if ( v >= vMaxThresh ) {
            spikeStatus = true;
            if ( tOut >= 0 ) {
                a += exp((tOut - time) / tau);
            } else {
                a += 1;
            }
            tOut = time;
            v -= vMaxThresh;
        }
    }
    isConsistent = true;
    return spikeStatus;
}

void LifNeuron::AddSynapse( const Synapse &synapse ) {
    outputSynapses.push_back( synapse );
}

void LifNeuron::Backward(float sumA) {
    float totalStrength = 0;
    size_t n = outputSynapses.size();
    for ( auto synapse: outputSynapses ) {
        if ( synapse.updatable ) {
            totalStrength += synapse.strength;
        }
    }
    grad = 0;
    for ( auto synapse: outputSynapses ) {
        if ( !(synapse.updatable) ) {
            continue;
        }
        float VMaxNext = synapse.next->vMaxThresh;
        synapse.DaDx = ( synapse.strength + sigma_mu * totalStrength / (1 - sigma_mu * (n - 1) )
                ) / (1 + sigma_mu) / VMaxNext;
        grad += synapse.DaDx * synapse.next->grad;
        synapse.DlDw = synapse.next->grad * a / exp (1 / tau) / VMaxNext;
    }
    DlDV = grad * (-(1 + sigma_mu) * a + sigma_mu * sumA) / vMaxThresh;

}

LifNeuron::LifNeuron() {
    outputSynapses = std::vector<Synapse>();
    v = 0;
    a= 0;
    vMaxThresh = 10;
    tau = 20;
    tps = -1;
    tOut = -1;
    tRef = 1;
    isConsistent = true;
    grad = 0;
    sigma_mu = 0;
    DlDV = 0;
    spikeCounter = 0;
}

void LifNeuron::Reset() {
    v = 0;
    a = 0;
    tps = -1;
    tOut = -1;
    isConsistent = true;
    grad = 0;
    DlDV = 0;
    spikeCounter = 0;
}
