#include <iostream>

#include "BasicNetwork.hpp"

#include "LifNeuron.h"
#include "Synapse.h"
#include "PreciseEventManager.h"

const int INPUT_SIZE = 1;
const int OUTPUT_SIZE = 4;


int main() {

    auto model = BasicNetwork<LifNeuron, Synapse, PreciseEventManager>( INPUT_SIZE, OUTPUT_SIZE, true );

    model.AddLink( 0, 1, 100, 0.1 );
    model.AddLink( 0, 2, 100, 0.15 );
    model.AddLink( 0, 3, 100, 0.05 );
    model.AddLink( 0, 4, 100, 0.2 );

    std::vector<float> output( OUTPUT_SIZE );

    model.Forward( {0.5, 100, 100, 100}, output, 4 );

    for ( auto activation: output ) {
        std::cout << activation << " ";
    }
    std::cout << "\n";
}