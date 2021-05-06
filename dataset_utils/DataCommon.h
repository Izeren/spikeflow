#pragma once

#include<random>
#include <SpikingGeneral.h>

namespace DATA_CONVERSION {

    void ConvertDataToTimings( SPIKING_NN::Dataset &data, float inputSimulationTime = 25 );

    void ConvertDataToSpikeTrains( SPIKING_NN::Dataset &rawData, SPIKING_NN::SpikeTrainDataset &data,
                                   std::default_random_engine &generator, float inputSimulationTime = 25 );

    void ConvertDataToUniformSpikeTrains( SPIKING_NN::Dataset &rawData, SPIKING_NN::SpikeTrainDataset &data,
                                          float inputSimulationTime = 25, float minRate = 0.2, float maxRate = 0.8 );

    void ConvertSamplesToTimings( std::vector<SPIKING_NN::Sample> &samples, float inputSimulationTime );

    void ConvertSamplesToSpikeTrains( std::vector<SPIKING_NN::Sample> &samples,
                                      std::vector<SPIKING_NN::SpikeTrain> &spikeTrains,
                                      float inputSimulationTime,
                                      std::default_random_engine &generator );

    void ConvertSamplesToUniformSpikeTrains(
            std::vector<SPIKING_NN::Sample> &samples,
            std::vector<SPIKING_NN::SpikeTrain> &spikeTrains,
            float inputSimulationTime,
            float minRate, float maxRate );
}
