#include "VanillaSynapseBuilder.h"
#include "BasicSynapse.h"

ISynapse *VanillaSynapseBuilder::Build( size_t layerSize, size_t nextLayerSize, INeuron *prev, INeuron *next ) const
{
//    3 stands for some fixed constant from the article
    float limit = sqrt( 3.f / ( static_cast<float>( nextLayerSize )));
    std::uniform_real_distribution<float> distribution( 0, limit );
    return new BasicSynapse(
            ISynapse::DEFAULT_SYNAPSE_UPDATABILITY,
            distribution( generator ),
            ISynapse::DEFAULT_SYNAPSE_DELAY,
            prev, next );
}

VanillaSynapseBuilder::VanillaSynapseBuilder( std::default_random_engine &generator ) : ISynapseBuilder( generator ) { }
