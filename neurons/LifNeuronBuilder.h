#pragma once

#include "INeuronBuilder.h"

class LifNeuronBuilder : public INeuronBuilder {
public:
    INeuron *build() const override;
};