#include <SpikingGeneral.h>
#include <fstream>
#include <algorithm>
#include "DigitsUtils.h"

void ReadSingleFileWithDigits( const std::string &path, std::vector<SPIKING_NN::Sample> &samples,
                               std::vector<float> &targets )
{
    std::ifstream file( path );
    if ( file.is_open()) {
        std::string line;
        while ( std::getline( file, line, '\n' )) {
            std::stringstream ss;
            ss << line;
            std::string param;
            std::vector<float> params;
            while ( std::getline( ss, param, ',' )) {
                params.push_back( std::stof( param ));
            }
            float label = params.back();
            params.pop_back();
            samples.push_back( params );
            targets.push_back( label );
        }
    }
}

void
DigitsUtils::ReadDigits( const std::string &trainPath, const std::string &valPath, SPIKING_NN::Dataset &dataset ) const
{
    ReadSingleFileWithDigits( trainPath, dataset.xTrain, dataset.yTrain );
    ReadSingleFileWithDigits( valPath, dataset.xTest, dataset.yTest );
}
