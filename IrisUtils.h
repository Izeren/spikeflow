#pragma once
#include <fstream>
#include <vector>
#include <random>
#include <string>
#include <sstream>
#include <cstring>
#include <algorithm>

namespace IRIS {

    typedef std::vector<std::vector<float > > Samples;
    typedef std::vector<int> Labels;

    typedef struct {
        Samples xTrain;
        Labels yTrain;
        Samples xTest;
        Labels yTest;
    } Dataset;

    void ReadIris( const std::string &path, Dataset &dataset );

    std::vector<int> ConvertSampleToSingularSpikes( const std::vector<float> &inputs );

    void SaveSplit( const std::string &path, Dataset &dataset );

};