#include "DenseLifLayer.h"
#include "ISynapse.h"

DenseLifLayer::DenseLifLayer( LayerMeta meta )
        : ILayer( std::move( meta )) { }

void DenseLifLayer::Init( size_t nextLayerSize )
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
    for ( auto neuron : neurons ) {
        size_t s = GetSize();
        neuron->RandomInit( meta.alpha, GetSize(), nextLayerSize );
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
    for ( auto neuron : neurons ) {
        stats.neuronMP.Add( neuron->GetMaxMP());
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

ILayer &DenseLifLayer::GradStep( size_t batchSize, float learningRateV, float learningRateW, float BETA, bool isInput )
{
    float N = GetSize(), M = 0, m = 0;
//    int N2 = layer[0].get()->outputSynapses.size();
//    if ( N2 == 0 ) {
//        N2 = 1;
//    }
    float S = 0;
    for ( auto neuron: neurons ) {
        for ( auto synapse: neuron->GetOutputSynapses()) {
            if ( synapse ) {
                if ( neuron->GetOutput() > 0 ) {
                    m += 1;
                    S += synapse->GetStrength() * synapse->GetStrength();
                }
                M += 1;
            }
        }
    }
    S = BETA * ( S - 1 );
    float F = exp( S );
    if ( m == 0 ) {
        m = 1;
    }
    if ( M == 0 ) {
        M = 1;
    }
//    int M = 1, N = 1, m = 1;
    auto batchFSize = static_cast<float>( batchSize );
    for ( auto neuron: neurons ) {
        // TODO: fix to real DLDV grad
        stats.gradV.Add( neuron->GetGrad() / batchFSize );
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

ILayer &DenseLifLayer::Backward( const std::vector<float> &deltas )
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
        neurons[idx]->Backward( totalLayerOutput, delta );
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
