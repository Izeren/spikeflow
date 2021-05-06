#pragma once

#include <random>
#include "ISynapseBuilder.h"

class VanillaSynapseBuilder : public ISynapseBuilder {
public:
    explicit VanillaSynapseBuilder( std::default_random_engine &generator );

    ~VanillaSynapseBuilder() override = default;

    ISynapse *build( size_t layerSize, size_t nextLayerSize, INeuron *prev, INeuron *next ) override;

};