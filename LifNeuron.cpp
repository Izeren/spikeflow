#include "LifNeuron.h"
#include "Synapse.h"
#include <iostream>

float LifNeuron::GetWDyn(float tOut, float tp, float tRef) {
    if (tOut < 0 || tp < 0 || tRef < 0 || tp - tOut >= tRef) {
        return 1;
    } else {
        return (tp - tOut) * (tp - tOut) / tRef / tRef;
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
//    std::cout << tOut << " " << time << " " << tRef << " " << wDyn << " wdyn \n";
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
            RelaxOutput( time, true );
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
            totalStrength += synapse.strength / synapse.next->vMaxThresh;
        }
    }
    grad = 0;
    for ( int synapseId = 0; synapseId < outputSynapses.size(); synapseId++ ) {
        if ( !(outputSynapses[synapseId].updatable) ) {
            continue;
        }
        outputSynapses[synapseId].DaDx = ( outputSynapses[synapseId].strength + sigma_mu * totalStrength / (1 - sigma_mu * (n - 1) )
                ) / (1 + sigma_mu);
        grad += outputSynapses[synapseId].next->grad * outputSynapses[synapseId].strength / vMaxThresh;
        outputSynapses[synapseId].DlDw = outputSynapses[synapseId].next->grad * a / exp (1 / tau);
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
    tRef = 3;
    isConsistent = true;
    grad = 0;
    sigma_mu = 0.025;
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

void LifNeuron::RelaxOutput( int time, bool withSpike) {
    if ( tOut >= 0 ) {
        a = a * exp((tOut - time) / tau) + withSpike;
    } else {
        a = withSpike;
    }
}
