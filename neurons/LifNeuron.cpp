#include <random>
#include "LifNeuron.h"
#include "LifSynapse.h"


float LifNeuron::GetWDyn( float tOut, float tp, float tRef )
{
    if ( tOut < 0 || tp < 0 || tRef < 0 || tp - tOut >= tRef ) {
        return 1;
    } else {
        return ( tp - tOut ) * ( tp - tOut ) / tRef / tRef;
    }
}


void LifNeuron::ProcessInputSpike( float time, float potential )
{
    float wDyn = GetWDyn( tOut, time, tRef );
    float leackyFactor;
    if ( tps < 0 ) {
        leackyFactor = 1;
    } else {
        leackyFactor = exp(( tps - time ) / tau );
    }

    this->potential = this->potential * leackyFactor + potential * wDyn;
    tps = time;
    inputSpikeCounter += 1;
    if ( this->potential < -this->vMaxThresh ) {
        this->potential = -this->vMaxThresh;
    }
    this->consistent = ( this->potential < this->vMaxThresh );
    outputSpikeCounter += !consistent;
}

LifNeuron::LifNeuron( float v, float a, float vMinThresh,
                      float vMaxThresh, float tau, float tps, float tOut, float tRef, bool isConsistent )
        : INeuron( v, tRef, isConsistent ), a( a ), vMaxThresh( vMaxThresh ),
          tau( tau ), tps( tps ), tOut( tOut )
{
    grad = 0;
    sigma_mu = 0.25;
    DlDV = 0;
}

bool LifNeuron::IsConsistent()
{
    return consistent;
}

void LifNeuron::NormalizePotential( float time )
{
    if ( potential >= vMaxThresh ) {
        RelaxOutput( time, true );
        potential -= vMaxThresh;
    }
    consistent = ( potential < vMaxThresh );
}


void LifNeuron::Backward( float sumOutput, float delta )
{
//    float totalStrength = 0;
//    for ( auto synapse: outputSynapses ) {
//        if ( synapse->IsUpdatable()) {
//            totalStrength += synapse->GetStrength() /
//                             dynamic_cast<LifNeuron *>(synapse->GetPostSynapticNeuron())->vMaxThresh;
//        }
//    }
    grad = outputSynapses.empty() ? delta : 0;
    for ( ISynapse *outputSynapse: outputSynapses ) {
        auto next = outputSynapse->GetPostSynapticNeuron();
        grad += next->GetGrad() * outputSynapse->GetStrength();
        if ( outputSynapse->IsUpdatable()) {
            outputSynapse->Backward( a );
        }
    }
    // The right side grad delta is used
    DlDV = grad * ( -( 1 + sigma_mu ) * a + sigma_mu * sumOutput ) / vMaxThresh;
    // This is for transferring grad delta from right side of the neuron to the left one
    grad /= vMaxThresh;
    batchDlDV += DlDV;
}

LifNeuron::LifNeuron()
{
    outputSynapses = std::unordered_set<ISynapse *>();
    inputSynapses = std::unordered_set<ISynapse *>();
    potential = 0;
    a = 0;
    vMaxThresh = 10;
    tau = 20;
    tps = -1;
    tOut = -1;
    fts = -1;
    tRef = 3.2412423543293534542952;
    consistent = true;
    grad = 0;
    sigma_mu = 0;
    DlDV = 0;
    batchDlDV = 0;
    inputSpikeCounter = 0;
    outputSpikeCounter = 0;
}

void LifNeuron::Reset()
{
    potential = 0;
    a = 0;
    tps = -1;
    tOut = -1;
    fts = -1;
    consistent = true;
    inputSpikeCounter = 0;
    outputSpikeCounter = 0;
}

void LifNeuron::ResetGrad()
{
    grad = 0;
    DlDV = 0;
    batchDlDV = 0;
}

void LifNeuron::RelaxOutput( float time, bool withSpike )
{
    if ( tOut >= 0 ) {
        a = a * exp(( tOut - time ) / tau ) + withSpike;
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

SPIKING_NN::Time LifNeuron::GetFirstSpikeTS()
{
    return fts;
}

float LifNeuron::GetOutput() const
{
    return a;
}

void LifNeuron::SetDlDV( float _gradV )
{
    DlDV = _gradV;
}

void LifNeuron::SetGrad( float _grad )
{
    grad = _grad;
}

float LifNeuron::GetGrad() const
{
    return grad;
}

float LifNeuron::GetDlDv() const
{
    return DlDV;
}

SPIKING_NN::Time LifNeuron::GetTau() const
{
    return tau;
}

float LifNeuron::GetSigmaMu() const
{
    return sigma_mu;
}

void LifNeuron::GradStep( float learningRate )
{
    vMaxThresh -= batchDlDV * learningRate; //* sqrt(N / M / m);
    //TODO: those neurons should be carefully studied
    if ( vMaxThresh < 0 ) {
        vMaxThresh = 0.001;
    }
}

void LifNeuron::SetVMaxThresh( float _vMaxTrhesh )
{
    vMaxThresh = _vMaxTrhesh;
}

void LifNeuron::SetSigmaMu( float sigmaMu )
{
    sigma_mu = sigmaMu;
}

void LifNeuron::RandomInit( float alpha, size_t layerSize, size_t nextLayerSize )
{
    // In fact, for the output layer (nextLayerSize == 0) we don't care about vMaxThresh. We asses only membrane potentials
    vMaxThresh = alpha * sqrt( 3. / ( nextLayerSize ? nextLayerSize : 1 ));
}

float LifNeuron::GetMaxMP()
{
    return vMaxThresh;
}

