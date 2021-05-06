#pragma once

#include <string>
#include "SpikingGeneral.h"

namespace IRIS {

    void ReadIris( const std::string &path, SPIKING_NN::Dataset &dataset, float valProbability );

    void SaveSplit( const std::string &path, SPIKING_NN::Dataset &dataset );

};