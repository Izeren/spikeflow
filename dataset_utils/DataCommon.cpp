#include "DataCommon.h"

void DATA_CONVERSION::ConvertSamplesToTimings( std::vector<SPIKING_NN::Sample> &samples, float inputSimulationTime )
{
    if ( samples.empty()) {
        return;
    }
    std::vector<float> minValues( samples[0].size(), 0 );
    std::vector<float> maxValues( samples[0].size(), 0 );
    for ( auto activationId = 0; activationId < samples[0].size(); ++activationId ) {
        minValues[activationId] = samples[0][activationId];
        maxValues[activationId] = samples[0][activationId];
    }

    for ( auto sampleId = 0; sampleId < samples.size(); ++sampleId ) {
        for ( auto activationId = 0; activationId < samples[0].size(); ++activationId ) {
            if ( samples[sampleId][activationId] < minValues[activationId] ) {
                minValues[activationId] = samples[sampleId][activationId];
            }
            if ( samples[sampleId][activationId] > maxValues[activationId] ) {
                maxValues[activationId] = samples[sampleId][activationId];
            }
        }
    }

    // We are not checking maxValues[idx] == minValues[idx] because such params are redundant in dataset
    for ( auto activationId = 0; activationId < samples[0].size(); ++activationId ) {
        float maxDelta = ( maxValues[activationId] - minValues[activationId] );
        for ( auto &sample : samples ) {
            float &activation = sample[activationId];
            float intensity = ( activation - minValues[activationId] ) / maxDelta;
            activation = ( 1 - intensity ) * inputSimulationTime;
        }
    }
}

void DATA_CONVERSION::ConvertSamplesToSpikeTrains(
        std::vector<SPIKING_NN::Sample> &samples,
        std::vector<SPIKING_NN::SpikeTrain> &spikeTrains,
        float inputSimulationTime,
        std::default_random_engine &generator )
{
    if ( samples.empty()) {
        return;
    }

    std::vector<float> minValues( samples[0].size(), 0 );
    std::vector<float> maxValues( samples[0].size(), 0 );
    for ( auto activationId = 0; activationId < samples[0].size(); ++activationId ) {
        minValues[activationId] = samples[0][activationId];
        maxValues[activationId] = samples[0][activationId];
    }

    for ( auto sampleId = 0; sampleId < samples.size(); ++sampleId ) {
        for ( auto activationId = 0; activationId < samples[0].size(); ++activationId ) {
            if ( samples[sampleId][activationId] < minValues[activationId] ) {
                minValues[activationId] = samples[sampleId][activationId];
            }
            if ( samples[sampleId][activationId] > maxValues[activationId] ) {
                maxValues[activationId] = samples[sampleId][activationId];
            }
        }
    }

    // We are not checking maxValues[idx] == minValues[idx] because such params are redundant in dataset
    std::uniform_real_distribution<float> distribution( 0, 1 );
    std::uniform_real_distribution<float> timeNoiseDistribution( 0, SPIKING_NN::TIME_STEP / 1e4f );
    size_t inputSize = samples[0].size();
    spikeTrains.resize( samples.size());
    for ( int idx = 0; idx < samples.size(); ++idx ) {
        spikeTrains[idx].resize( inputSize );
    }
    for ( auto activationId = 0; activationId < inputSize; ++activationId ) {
        float maxDelta = ( maxValues[activationId] - minValues[activationId] );
        for ( int sid = 0; sid < samples.size(); ++sid ) {
            float &activation = samples[sid][activationId];
            float intensity = ( activation - minValues[activationId] ) / maxDelta;
            for ( int t = 0; t < inputSimulationTime; ++t ) {
                if ( distribution( generator ) < intensity ) {
                    spikeTrains[sid][activationId].emplace_back((float) t + timeNoiseDistribution( generator ));
                }
            }
        }
    }
}

void DATA_CONVERSION::ConvertSamplesToUniformSpikeTrains(
        std::vector<SPIKING_NN::Sample> &samples,
        std::vector<SPIKING_NN::SpikeTrain> &spikeTrains,
        float inputSimulationTime,
        float minRate, float maxRate )
{
    if ( samples.empty()) {
        return;
    }

    std::vector<float> minValues( samples[0].size(), 0 );
    std::vector<float> maxValues( samples[0].size(), 0 );
    for ( auto activationId = 0; activationId < samples[0].size(); ++activationId ) {
        minValues[activationId] = samples[0][activationId];
        maxValues[activationId] = samples[0][activationId];
    }

    for ( auto sampleId = 0; sampleId < samples.size(); ++sampleId ) {
        for ( auto activationId = 0; activationId < samples[0].size(); ++activationId ) {
            if ( samples[sampleId][activationId] < minValues[activationId] ) {
                minValues[activationId] = samples[sampleId][activationId];
            }
            if ( samples[sampleId][activationId] > maxValues[activationId] ) {
                maxValues[activationId] = samples[sampleId][activationId];
            }
        }
    }

    // We are not checking maxValues[idx] == minValues[idx] because such params are redundant in dataset
    std::uniform_real_distribution<float> distribution( 0, 1 );
    std::uniform_real_distribution<float> timeNoiseDistribution( 0, SPIKING_NN::TIME_STEP / 1e4f );
    size_t inputSize = samples[0].size();
    spikeTrains.resize( samples.size());
    for ( int idx = 0; idx < samples.size(); ++idx ) {
        spikeTrains[idx].resize( inputSize );
    }
    for ( auto activationId = 0; activationId < inputSize; ++activationId ) {
        float maxDelta = ( maxValues[activationId] - minValues[activationId] );
        for ( int sid = 0; sid < samples.size(); ++sid ) {
            float &activation = samples[sid][activationId];
            float intensity = ( activation - minValues[activationId] ) / maxDelta;
            int numSpikes = static_cast<int>(inputSimulationTime * ( intensity * ( maxRate - minRate ) + minRate ));
            float spikeStep = inputSimulationTime / numSpikes;
            for ( int tdx = 0; tdx < numSpikes; ++tdx ) {
                spikeTrains[sid][activationId].emplace_back( tdx * spikeStep );
            }
        }
    }
}

void DATA_CONVERSION::ConvertDataToTimings( SPIKING_NN::Dataset &data, float inputSimulationTime )
{
    ConvertSamplesToTimings( data.xTrain, inputSimulationTime );
    ConvertSamplesToTimings( data.xTest, inputSimulationTime );
}

void DATA_CONVERSION::ConvertDataToSpikeTrains( SPIKING_NN::Dataset &rawData, SPIKING_NN::SpikeTrainDataset &data,
                                                std::default_random_engine &generator,
                                                float inputSimulationTime )
{
    data.yTrain = rawData.yTrain;
    data.yTest = rawData.yTest;
    ConvertSamplesToSpikeTrains( rawData.xTrain, data.xTrain, inputSimulationTime, generator );
    ConvertSamplesToSpikeTrains( rawData.xTest, data.xTest, inputSimulationTime, generator );
}

void
DATA_CONVERSION::ConvertDataToUniformSpikeTrains( SPIKING_NN::Dataset &rawData, SPIKING_NN::SpikeTrainDataset &data,
                                                  float inputSimulationTime, float minRate, float maxRate )
{
    data.yTrain = rawData.yTrain;
    data.yTest = rawData.yTest;
    ConvertSamplesToUniformSpikeTrains( rawData.xTrain, data.xTrain, inputSimulationTime, minRate, maxRate );
    ConvertSamplesToUniformSpikeTrains( rawData.xTest, data.xTest, inputSimulationTime, minRate, maxRate );
}
