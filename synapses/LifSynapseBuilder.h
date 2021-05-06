#pragma once

#include "ISynapseBuilder.h"

class LifSynapseBuilder : public ISynapseBuilder {

public:
    explicit LifSynapseBuilder( std::default_random_engine &generator );

    ~LifSynapseBuilder() override = default;

    ISynapse *build( size_t layerSize, size_t nextLayerSize, INeuron *prev, INeuron *next ) override;

};