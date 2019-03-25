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

    std::vector<int> ConvertSampleToSingularSpikes( const std::vector<float> &inputs );

    void SaveSplit( const std::string &path, SPIKING_NN::Dataset &dataset );

};