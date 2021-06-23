#pragma once

class DemoScripts {
public:
    void TrainVanillaIris( char *path, std::default_random_engine &generator );

    void TrainSpikingIris( char *path, std::default_random_engine &generator );

    void RunDummyModel();

    void TrainSpikingDigits( const char *trainPath, const char *valPath, std::default_random_engine &generator, const
    char *logsPath );
};