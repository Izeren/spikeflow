#pragma once

#include <IEventManager.h>
#include "../layers/ILayer.h"
#include "../layers/LayerMeta.h"
#include "../layers/ILayerBuilder.h"

class IDenseNetwork {
public:
    explicit IDenseNetwork( std::vector<ILayer *> _layers, IEventManager &_eventManager,
                            size_t sInConnections, size_t sBetweenConnections );

    IDenseNetwork &Relax( SPIKING_NN::Time time );

    IDenseNetwork &LogBasicStats();

    IDenseNetwork &Backward( const std::vector<float> &deltas );

    IDenseNetwork &GradStep( size_t batchSize, float learningRateV, float learningRateW, float beta, float lambda );

    IDenseNetwork &Reset();

    IDenseNetwork &ResetStats();

    std::string GetStringStats() const;

    std::vector<float> Forward( const SPIKING_NN::SpikeTrain &sample, float simulationTime, bool useStdp );

    std::vector<INeuron *> GetOutputNeurons();

    size_t sInConnections;

    size_t sBetweenConnections;

private:
    std::vector<ILayer *> layers;
    IEventManager &eventManager;
};

class IDenseNetworkBuilder {
public:
    IDenseNetwork *
    Build( const std::vector<LayerMeta> &layersMeta, const ISynapseBuilder &synapseBuilder,
           const ILayerBuilder &layerBuilder, IEventManager &eventManager,
           std::default_random_engine &generator, float induceDistLimit ) const;
};