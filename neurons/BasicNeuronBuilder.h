#pragma once

#include <INeuron.h>
#include "INeuronBuilder.h"

class BasicNeuronBuilder : public INeuronBuilder {
public:
    INeuron *build() const override;
};