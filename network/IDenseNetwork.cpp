#include <PreciseEventManager.h>
#include "IDenseNetwork.h"

IDenseNetwork &IDenseNetwork::Relax( SPIKING_NN::Time time )
{
    for ( auto layer: layers ) {
        layer->Relax( time );
    }
    return *this;
}

IDenseNetwork &IDenseNetwork::LogBasicStats()
{
    for ( auto layer: layers ) {
        layer->LogBasicStats();
    }
    return *this;
}

IDenseNetwork::IDenseNetwork( std::vector<ILayer *> _layers, IEventManager &_eventManager ) : layers( std::move(
        _layers )), eventManager( _eventManager ) { }

IDenseNetwork &IDenseNetwork::Backward( const std::vector<float> &deltas )
{
    layers.back()->Backward( deltas );
    for ( auto it = layers.rbegin() + 1; it != layers.rend(); ++it ) {
        ( *it )->Backward( {} );
    }
    return *this;
}

IDenseNetwork &IDenseNetwork::GradStep( size_t batchSize, float learningRateV, float learningRateW, float beta )
{
    for ( size_t ldx = layers.size() - 1; ldx > 0; --ldx ) {
        layers[ldx]->GradStep( batchSize, learningRateV, learningRateW, beta, false );
    }
    layers.front()->GradStep( batchSize, learningRateV, learningRateW, beta, true );
    for ( auto layer: layers ) {
        layer->ResetGrad();
    }
    return *this;
}

IDenseNetwork &IDenseNetwork::Reset()
{
    for ( auto layer: layers ) {
        layer->ResetPotentials();
    }
    return *this;
}

IDenseNetwork &IDenseNetwork::ResetStats()
{
    for ( auto layer: layers ) {
        layer->ResetStats();
    }
    return *this;
}

std::string IDenseNetwork::GetStringStats() const
{
    std::stringstream ss;
    for ( ILayer *layer: layers ) {
        ss << *layer << "\n";
    }
    return ss.str();
}

std::vector<float> IDenseNetwork::Forward( const SPIKING_NN::SpikeTrain &sample, float simulationTime, bool useStdp )
{
    ILayer &output = *layers.back();
    eventManager.RegisterSpikeTrain( sample, *layers.front());
    eventManager.RunSimulation( simulationTime, useStdp );
    Relax( simulationTime ).LogBasicStats();

    auto result = std::vector<float>( layers.back()->GetSize());
    for ( size_t idx = 0; idx < result.size(); ++idx ) {
        result[idx] = output[idx]->GetOutput();
    }
    return result;
}

float GetDist( INeuron &n1, INeuron &n2 )
{
    return ( n1.x - n2.x ) * ( n1.x - n2.x )
           + ( n1.y - n2.y ) * ( n1.y - n2.y )
           + ( n1.z - n2.z ) * ( n1.z - n2.z );
}

void BuildSpatialConnections( std::vector<ILayer *> &layers, float distLimitSquared )
{
    for ( size_t l1dx = 0; l1dx < layers.size(); ++l1dx ) {
        for ( size_t l2dx = 0; l2dx < layers.size(); ++l2dx ) {
            for ( size_t n1dx = 0; n1dx < layers[l1dx]->neurons.size(); ++n1dx ) {
                for ( size_t n2dx = 0; n2dx < layers[l2dx]->neurons.size(); ++n2dx ) {
                    if ( l1dx == l2dx && n1dx == n2dx ) {
                        continue;
                    }
                    INeuron *n1 = layers[l1dx]->neurons[n1dx];
                    INeuron *n2 = layers[l2dx]->neurons[n2dx];
                    float dist = GetDist( *n1, *n2 );
                    if ( dist < distLimitSquared ) {
                        n1->neighbours.insert( std::make_pair( dist, n2 ));
                        n2->neighbours.insert( std::make_pair( dist, n1 ));
                    }
                }
            }
        }
    }
}

IDenseNetwork *
IDenseNetworkBuilder::Build( const std::vector<LayerMeta> &layersMeta, const ISynapseBuilder &synapseBuilder,
                             const ILayerBuilder &layerBuilder, IEventManager &eventManager,
                             std::default_random_engine &generator, float induceDistLimit ) const
{
    std::vector<ILayer *> layers;
    layers.resize( layersMeta.size());
    for ( int ldx = 0; ldx < layersMeta.size(); ++ldx ) {
        layers[ldx] = layerBuilder.Build( layersMeta[ldx] );
    }
    float zShift = 0;

    for ( size_t ldx = 0; ldx < layers.size(); ++ldx ) {
        size_t nextLayerSize = ldx + 1 < layers.size() ? layers[ldx + 1]->GetSize() : 0;
        zShift += layers[ldx]->GetZShift();
        layers[ldx]->Init( nextLayerSize, generator, zShift );
        if ( nextLayerSize ) {
            layers[ldx]->BindWithNext( *layers[ldx + 1], synapseBuilder );
        }
    }
    BuildSpatialConnections( layers, induceDistLimit );
    return new IDenseNetwork( layers, eventManager );
}
