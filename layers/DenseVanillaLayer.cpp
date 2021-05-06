#include "DenseVanillaLayer.h"
#include "ISynapse.h"

DenseVanillaLayer::DenseVanillaLayer( std::string _name, size_t _size, const INeuronBuilder &_neuronBuilder ) : ILayer(
        std::move( _name ), _size, _neuronBuilder ) { }

void DenseVanillaLayer::Init( float alpha, size_t nextLayerSize )
{
    for ( auto neuron : neurons ) {
        neuron->RandomInit( alpha, size, nextLayerSize );
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
        stats.neuronMP.Add( neuron->GetMaxMP());
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
DenseVanillaLayer::GradStep( size_t batchSize, float learningRateV, float learningRateW, float BETA, bool isInput )
{
    auto batchFSize = static_cast<float>( batchSize );
    for ( auto neuron: neurons ) {
        neuron->GradStep( isInput ? 0 : learningRateV / batchFSize );
        const ISynapses &synapses = neuron->GetOutputSynapses();
        for ( auto synapse: synapses ) {
            if ( synapse->IsUpdatable()) {
                stats.gradW.Add( synapse->GetGrad() / batchFSize );
                synapse->GradStep( learningRateW / batchFSize );
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
    for ( int idx = 0; idx < size; ++idx ) {
        float delta = 0;
        if ( !deltas.empty()) {
            delta = deltas[idx];
        }
        neurons[idx]->Backward( totalLayerOutput, delta );
    }
    return *this;
}

std::string DenseVanillaLayer::ToString() const
{
    return std::string();
}

