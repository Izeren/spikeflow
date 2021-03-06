#include "DenseVanillaLayer.h"
#include "ISynapse.h"

DenseVanillaLayer::DenseVanillaLayer( LayerMeta meta ) : ILayer( std::move( meta )) { }

void DenseVanillaLayer::Init( size_t nextLayerSize, std::default_random_engine &generator, float z )
{
    std::uniform_real_distribution<float> dist;
    for ( auto neuron : neurons ) {
        neuron->RandomInit( meta.alpha, GetSize(), nextLayerSize, 0, dist, generator );
    }
}

ILayer &DenseVanillaLayer::Relax( SPIKING_NN::Time time )
{
    for ( auto neuron : neurons ) {
        neuron->RelaxOutput( time, false );
    }
    return *this;
}

ILayer &DenseVanillaLayer::LogBasicStats()
{
    for ( auto neuron : neurons ) {
        stats.neuronMPMax.Add( neuron->GetMaxMP());
        for ( auto synapse : neuron->GetOutputSynapses()) {
            stats.synapseWeight.Add( synapse->GetStrength());
        }
        stats.spikes.Add( neuron->GetOutputSpikeCounter());
    }
    return *this;
}

ILayer &DenseVanillaLayer::ResetStats()
{
    stats.Reset();
    return *this;
}

ILayer &DenseVanillaLayer::ResetPotentials()
{
    for ( auto neuron: neurons ) {
        neuron->Reset();
    }
    return *this;
}

ILayer &DenseVanillaLayer::ResetGrad()
{
    for ( auto neuron: neurons ) {
        neuron->ResetGrad();
    }
    return *this;
}

ILayer &
DenseVanillaLayer::GradStep( size_t batchSize, float learningRateV, float learningRateW, float BETA, bool isInput,
                             float LAMBDA, bool isOutput )
{
    auto batchFSize = static_cast<float>( batchSize );
    for ( auto neuron: neurons ) {
        neuron->GradStep( isInput ? 0 : learningRateV / batchFSize, 0, 0, 0 );
        const ISynapses &synapses = neuron->GetOutputSynapses();
        for ( auto synapse: synapses ) {
            if ( synapse->IsUpdatable()) {
                stats.gradW.Add( synapse->GetGrad() / batchFSize );
                synapse->GradStep( learningRateW / batchFSize, 0, 0, 0 );
            }
        }
    }
    return *this;
}

ILayer &DenseVanillaLayer::Backward( const std::vector<float> &deltas )
{
    float totalLayerOutput = 0;
    for ( auto neuron: neurons ) {
        totalLayerOutput += neuron->GetOutput();
    }
    for ( int idx = 0; idx < GetSize(); ++idx ) {
        float delta = 0;
        if ( !deltas.empty()) {
            delta = deltas[idx];
        }
        neurons[idx]->Backward( totalLayerOutput, delta, 0, 0, 0 );
    }
    return *this;
}

std::string DenseVanillaLayer::ToString() const
{
    return std::string();
}

ILayer &DenseVanillaLayer::Forward()
{
    std::vector<float> output = std::vector<float>( GetSize());
    for ( auto neuron: neurons ) {
        float inPotential = 0;
        for ( auto synapse: neuron->GetInputSynapses()) {
            inPotential += synapse->GetPreSynapticNeuron()->GetOutput() * synapse->GetStrength();
        }
        neuron->ProcessInputSpike( 0, inPotential );
    }
    return *this;
}

