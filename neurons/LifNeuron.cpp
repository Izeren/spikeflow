#include "LifNeuron.h"
#include "Synapse.h"


float LifNeuron::GetWDyn( float tOut, float tp, float tRef ) {
    if ( tOut < 0 || tp < 0 || tRef < 0 || tp - tOut >= tRef ) {
        return 1;
    } else {
        return (tp - tOut) * (tp - tOut) / tRef / tRef;
    }
}


void LifNeuron::ProcessInputSpike( float time, float potential ) {
    float wDyn = GetWDyn( tOut, time, tRef );
    float leackyFactor;
    if ( tps < 0 ) {
        leackyFactor = 1;
    } else {
        leackyFactor = exp((tps - time) / tau );
    }

    this->potential = this->potential * leackyFactor + potential * wDyn;
    tps = time;
    spikeCounter += 1;
    if (this->potential < -this->vMaxThresh) {
        this->potential = -this->vMaxThresh;
    }
    this->consistent = (this->potential < this->vMaxThresh);
}

LifNeuron::LifNeuron( float v, float a, float vMinThresh,
                      float vMaxThresh, float tau, float tps, float tOut, float tRef, bool isConsistent )
        : INeuron(v, tRef, isConsistent), a( a ), vMaxThresh( vMaxThresh ),
          tau( tau ), tps( tps ), tOut( tOut ) {
    grad = 0;
    sigma_mu = 0.25;
    DlDV = 0;
}

bool LifNeuron::IsConsistent() {
    return consistent;
}

void LifNeuron::NormalizePotential( float time ) {
    if ( potential >= vMaxThresh ) {
        RelaxOutput( time, true );
        potential -= vMaxThresh;
    }
    consistent = (potential < vMaxThresh);
}


void LifNeuron::Backward( float sumA ) {
    float totalStrength = 0;
    for ( auto synapse: outputSynapses ) {
        if ( dynamic_cast<Synapse *>(synapse)->IsUpdatable()) {
            totalStrength += synapse->GetStrength() /
                             dynamic_cast<LifNeuron *>(synapse->GetPostSynapticNeuron())->vMaxThresh;
        }
    }
    grad = 0;
    for ( ISynapse *outputSynapse: outputSynapses ) {
        auto outputSynapsePtr = dynamic_cast<Synapse *>(outputSynapse);
        if ( !(outputSynapsePtr->IsUpdatable())) {
            continue;
        }
        auto next = dynamic_cast<LifNeuron *>(outputSynapsePtr->GetPostSynapticNeuron());
        grad += next->grad * outputSynapsePtr->GetStrength();
        outputSynapsePtr->SetDlDw( next->grad * a );
    }
//    DlDV = grad * (-(1 + sigma_mu) * a + sigma_mu * sumA) / vMaxThresh;
    DlDV = 0;


}

LifNeuron::LifNeuron() {
    outputSynapses = std::unordered_set<ISynapse *>();
    inputSynapses = std::unordered_set<ISynapse *>();
    potential = 0;
    a = 0;
    vMaxThresh = 10;
    tau = 20;
    tps = -1;
    tOut = -1;
    fts = -1;
    tRef = 3;
    consistent = true;
    grad = 0;
    sigma_mu = 0.025;
    DlDV = 0;
    spikeCounter = 0;
}

void LifNeuron::Reset() {
    potential = 0;
    a = 0;
    tps = -1;
    tOut = -1;
    fts = -1;
    consistent = true;
    grad = 0;
    DlDV = 0;
    spikeCounter = 0;
}

void LifNeuron::RelaxOutput( float time, bool withSpike ) {
    if ( tOut >= 0 ) {
        a = a * exp((tOut - time) / tau ) + withSpike;
    } else {
        a = withSpike;
    }
    if ( withSpike ) {
        tOut = time;
        if ( fts < 0 ) {
            fts = time;
        }
    }
}

float LifNeuron::GetOutput() {
    if ( fts <= SPIKING_NN::EPS ) {
        return 0;
    } else {
        return 1 / fts;
    }
}
