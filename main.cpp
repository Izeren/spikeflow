#include <iostream>
#include <random>
#include <scripts/DemoScripts.h>

int main( int argc, char **argv )
{
    const int RANDOM_SEED = 42;
    std::default_random_engine GENERATOR( RANDOM_SEED );

    if ( argc == 1 ) {
    } else {
        DemoScripts().TrainSpikingIris( argv[1], GENERATOR );
    }
}