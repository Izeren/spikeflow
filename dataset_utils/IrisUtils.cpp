#include "IrisUtils.h"
#include <fstream>
#include <random>
#include <sstream>
#include <algorithm>

void IRIS::ReadIris(
        const std::string &path,
        SPIKING_NN::Dataset &dataset,
        float valProbability )
{
    std::ifstream file( path );
    std::vector<std::vector<std::string> > allData;
    if ( file.is_open()) {
        std::default_random_engine generator( clock());
        std::uniform_real_distribution<float> distribution( 0, 1 );
        std::string line;
        while ( std::getline( file, line, '\n' )) {
            std::stringstream ss;
            ss << line;
            std::string param;
            std::vector<std::string> params;
            while ( std::getline( ss, param, ',' )) {
                params.push_back( param );
            }
            allData.push_back( params );
        }
        std::random_shuffle( allData.begin(), allData.end());
        for ( auto params: allData ) {
            int target = 0;
            if ( params.back() == "Iris-versicolor" ) {
                target = 1;
            } else if ( params.back() == "Iris-virginica" ) {
                target = 2;
            }
            if ( distribution( generator ) < valProbability ) {
                dataset.xTest.emplace_back( std::vector<float>());
                dataset.yTest.emplace_back( target );
                for ( int paramId = 0; paramId < params.size() - 1; ++paramId ) {
                    dataset.xTest.back().emplace_back( static_cast<float>(atof( params[paramId].c_str())));
                }
            } else {
                dataset.xTrain.emplace_back( std::vector<float>());
                dataset.yTrain.emplace_back( target );
                for ( int paramId = 0; paramId < params.size() - 1; ++paramId ) {
                    dataset.xTrain.back().emplace_back( static_cast<float>(atof( params[paramId].c_str())));
                }
            }
        }
    }
}


void IRIS::SaveSplit( const std::string &path, SPIKING_NN::Dataset &dataset )
{
    std::ofstream file( path );
    file << dataset.xTrain.size() << "\n";
    for ( int sampleId = 0; sampleId < dataset.xTrain.size(); ++sampleId ) {
        for ( int paramId = 0; paramId < dataset.xTrain[0].size(); ++paramId ) {
            file << dataset.xTrain[sampleId][paramId] << " ";
        }
        file << dataset.yTrain[sampleId] << "\n";
    }
    file << dataset.xTest.size() << "\n";
    for ( int sampleId = 0; sampleId < dataset.xTest.size(); ++sampleId ) {
        for ( int paramId = 0; paramId < dataset.xTest[0].size(); ++paramId ) {
            file << dataset.xTest[sampleId][paramId] << " ";
        }
        file << dataset.yTest[sampleId] << "\n";
    }
}
