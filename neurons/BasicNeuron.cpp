#include <BasicSynapse.h>
#include "BasicNeuron.h"

float BasicNeuron::ProcessInputSpike( SPIKING_NN::Time time, SPIKING_NN::Potential _potential )
{
    potential = _potential;
}

void BasicNeuron::Reset()
{
    potential = 0;
    consistent = true;
    grad = 0;
    inputSpikeCounter = 0;
    outputSpikeCounter = 0;
    grad = 0;
}

void BasicNeuron::ResetGrad()
{
    grad = 0;
}

void BasicNeuron::RandomInit( float alpha, size_t layerSize, size_t nextLayerSize, float z, std::uniform_real_distribution<float> &dist,
                              std::default_random_engine &generator ) { }

BasicNeuron::BasicNeuron()
{
    outputSynapses = std::unordered_set<ISynapse *>();
    inputSynapses = std::unordered_set<ISynapse *>();
    potential = 0;
    tRef = 0;
    consistent = true;
    grad = 0;
    inputSpikeCounter = 0;
    outputSpikeCounter = 0;
}

void BasicNeuron::RelaxOutput( SPIKING_NN::Time time, bool withSpike ) { }

void BasicNeuron::Backward( float layerTotalOutput, float delta )
{
    if ( outputSynapses.empty()) {
        grad = delta;
    }
    for ( ISynapse *outputSynapse: outputSynapses ) {
        auto outputSynapsePtr = outputSynapse;
        if ( !( outputSynapsePtr->IsUpdatable())) {
            continue;
        }
        auto next = outputSynapsePtr->GetPostSynapticNeuron();
        grad += next->GetGrad() * outputSynapsePtr->GetStrength();
        outputSynapsePtr->Backward( potential );
    }
}

void BasicNeuron::SetGrad( float _grad )
{
    grad = _grad;
}

float BasicNeuron::GetGrad() const
{
    return grad;
}

void BasicNeuron::GradStep( float learningRate )
{
    // Does nothing cause we don't have neuron based dynamic parameters
}

float BasicNeuron::GetOutput() const
{
    return potential;
}

float BasicNeuron::NormalizePotential( SPIKING_NN::Time time )
{
    // Does nothing because this logic is not applicable
}

SPIKING_NN::Time BasicNeuron::GetFirstSpikeTS()
{
    // Returns -1 as spike logic is not applicable
    return -1;
}

float BasicNeuron::GetMaxMP()
{
    return 0;
}
