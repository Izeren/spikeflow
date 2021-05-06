#include <random>
#include "LifSynapseBuilder.h"
#include "LifSynapse.h"

ISynapse *LifSynapseBuilder::build( size_t layerSize, size_t nextLayerSize, INeuron *prev, INeuron *next )
{
//    3 stands for some fixed constant from the article
    float limit = sqrt( 3.f / ( static_cast<float>( nextLayerSize )));
    // Exps with positive synapses only ( something strange with average synapse strength )
    std::uniform_real_distribution<float> weightDistribution( -limit, limit );
    std::uniform_real_distribution<float> delayModulationDistribution( -SPIKING_NN::TIME_STEP / 1000,
                                                                       SPIKING_NN::TIME_STEP / 1000 );
    return new LifSynapse(
            ISynapse::DEFAULT_SYNAPSE_UPDATABILITY,
            weightDistribution( generator ),
            ISynapse::DEFAULT_SYNAPSE_DELAY,
            prev, next );
}

LifSynapseBuilder::LifSynapseBuilder( std::default_random_engine &generator ) : ISynapseBuilder( generator ) { }
