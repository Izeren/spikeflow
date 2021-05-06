#pragma once

#include <SpikingGeneral.h>

class ILossFunction {
public:
    SPIKING_NN::Score eval( std::vector<SPIKING_NN::Output> &predictions, std::vector<SPIKING_NN::Target> &labels );
};