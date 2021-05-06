#pragma once

#include <INeuron.h>

class INeuronBuilder {
public:
    virtual INeuron *build() const = 0;
};