#include <iostream>
#include <random>
#include <scripts/DemoScripts.h>

int main( int argc, char **argv )
{
    const int RANDOM_SEED = 42;
    std::default_random_engine GENERATOR( RANDOM_SEED );

    if ( argc == 1 ) {
    } else if ( argc == 3 ) {
        DemoScripts().TrainSpikingIris( argv[1], GENERATOR );
    } else {
        DemoScripts().TrainSpikingDigits( argv[1], argv[2], GENERATOR, argv[3] );
    }
}