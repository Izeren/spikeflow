#pragma once

#include <fstream>
#include <vector>
#include <random>
#include <string>
#include <sstream>
#include <cstring>
#include <algorithm>
#include "SpikingGeneral.h"

namespace IRIS {

    void ReadIris( const std::string &path, SPIKING_NN::Dataset &dataset );

    void ConvertIrisToTimings( SPIKING_NN::Dataset &data );

    void ConvertSamplesToTimings( std::vector<SPIKING_NN::Sample> &samples );

    void SaveSplit( const std::string &path, SPIKING_NN::Dataset &dataset );

};