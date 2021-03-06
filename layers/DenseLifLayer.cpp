#include "DenseLifLayer.h"
#include "ISynapse.h"

DenseLifLayer::DenseLifLayer( LayerMeta meta )
        : ILayer( std::move( meta )) { }

void DenseLifLayer::Init( size_t nextLayerSize, std::default_random_engine &generator, float z )
{
    // TODO: implement lateral inhibition as part of layer init
//void createLateralInhibitionSynapses( SPIKING_NN::Layer &layer, int size ) {
//    for ( int prevId = 0; prevId < size; prevId++ ) {
//        for ( int nextId = 0; nextId < size; nextId++ ) {
//            if ( prevId != nextId ) {
//                INeuron *prev = layer[prevId];
//                INeuron *next = layer[nextId];
//                ISynapse *synapsePtr = new LifSynapse(
//                        false,
//                        ISynapse::DEFAULT_LATERAL_SYNAPSE_STRENGTH,
//                        ISynapse::DEFAULT_SYNAPSE_DELAY, prev, next );
//                prev->AddOutputSynapse( synapsePtr );
//                next->AddInputSynapse( synapsePtr );
//            }
//        }
//    }
//}
    std::uniform_real_distribution<float> coordDist( -meta.width, meta.width );
    for ( auto neuron : neurons ) {
        neuron->RandomInit( meta.alpha, GetSize(), nextLayerSize, z, coordDist, generator );
    }
}

ILayer &DenseLifLayer::Relax( SPIKING_NN::Time time )
{
    for ( auto neuron : neurons ) {
        neuron->RelaxOutput( time, false );
    }
    return *this;
}

ILayer &DenseLifLayer::LogBasicStats()
{
    stats.synapseWeight.SetRef( 1.f / neurons.size());
    for ( auto neuron : neurons ) {
        stats.neuronMPMax.Add( neuron->GetMaxMP());
        stats.neuronMP.Add( neuron->GetPotential());
        stats.neuronMPInduced.Add( neuron->GetInduced());
        stats.inducedSpikes.Add( neuron->GetInductionAffectedSpikes());
        for ( auto synapse : neuron->GetOutputSynapses()) {
            stats.synapseWeight.Add( synapse->GetStrength());
        }
        stats.spikes.Add( neuron->GetOutputSpikeCounter());
    }
    return *this;
}

ILayer &DenseLifLayer::ResetStats()
{
    stats.Reset();
    return *this;
}

ILayer &DenseLifLayer::ResetPotentials()
{
    for ( auto neuron: neurons ) {
        neuron->Reset();
    }
    return *this;
}

ILayer &DenseLifLayer::ResetGrad()
{
    for ( auto neuron: neurons ) {
        neuron->ResetGrad();
    }
    return *this;
}

ILayer &DenseLifLayer::GradStep( size_t batchSize, float learningRateV, float learningRateW, float BETA, bool isInput,
                                 float LAMBDA, bool isOutput )
{
    float S = 0;
    for ( auto neuron: neurons ) {
        for ( auto synapse: neuron->GetOutputSynapses()) {
            if ( synapse ) {
                if ( neuron->GetOutput() > 0 ) {
                    S += synapse->GetStrength() * synapse->GetStrength();
                }
            }
        }
    }
    // Don't need to check for output layer. It doesn't have any weights
    float weightNormFactor = BETA * LAMBDA * exp( BETA * ( S / neurons.size() - 1. ));
    size_t activeNeurons = 0;
    for ( auto neuron: neurons ) {
        activeNeurons += neuron->GetOutputSpikeCounter() > 0;
    }
    auto batchFSize = static_cast<float>( batchSize );
    stats.gradW.SetRef( 1.f / neurons.size());
    stats.gradV.SetRef( 1.f / neurons.size());
    for ( auto neuron: neurons ) {
        // TODO: fix to real DLDV grad
        stats.gradV.Add( neuron->GetGrad() / batchFSize );
        size_t inputActiveSynapses = 0;
        for ( auto synapse: neuron->GetInputSynapses()) {
            inputActiveSynapses += synapse->GetPreSynapticNeuron()->GetOutputSpikeCounter() > 0;
        }
        neuron->GradStep(
                isInput ? 0 : learningRateV / batchFSize,
                neurons.size(),
                neuron->GetInputSynapses().size(),
                inputActiveSynapses );
        const ISynapses &synapses = neuron->GetOutputSynapses();
        for ( auto synapse: synapses ) {
            if ( synapse->IsUpdatable()) {
                stats.gradW.Add( synapse->GetGrad() / batchFSize );
                synapse->GradStep( learningRateW / batchFSize, activeNeurons, synapses.size(), weightNormFactor );
            }
        }
    }
    return *this;
}

ILayer &DenseLifLayer::Backward( const std::vector<float> &deltas )
{
    float totalLayerOutput = 0;
    size_t activeNeurons = 0;
    float meanReversedSquaredThresholds = 0;
    for ( auto neuron: neurons ) {
        totalLayerOutput += neuron->GetOutput();
        activeNeurons += neuron->GetOutputSpikeCounter() > 0;
        meanReversedSquaredThresholds += 1. / neuron->GetMaxMP() / neuron->GetMaxMP();
    }
    meanReversedSquaredThresholds = sqrt( 1. * meanReversedSquaredThresholds / neurons.size() );

    for ( int idx = 0; idx < GetSize(); ++idx ) {
        float delta = 0;
        if ( !deltas.empty()) {
            delta = deltas[idx];
        }
        neurons[idx]->Backward( totalLayerOutput, delta, meta.size, activeNeurons, meanReversedSquaredThresholds );
    }
    return *this;
}

std::string DenseLifLayer::ToString() const
{
    std::stringstream ss;
    ss << "Stats for layer '" << GetName() << "': " << stats;
    return ss.str();
}

ILayer &DenseLifLayer::Forward()
{
    return *this;
}
