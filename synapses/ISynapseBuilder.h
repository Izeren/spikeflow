#pragma once

#include <random>
#include <ISynapse.h>

class ISynapseBuilder {
public:
    explicit ISynapseBuilder( std::default_random_engine &generator );

    virtual ~ISynapseBuilder() = default;

    virtual ISynapse *build( size_t layerSize, size_t nextLayerSize, INeuron *prev, INeuron *next ) = 0;

protected:
    std::default_random_engine generator;
};
