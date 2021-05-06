#include "PreciseEventManager.h"
#include "INeuron.h"
#include "ISynapse.h"
#include <iostream>

/**
 *
 * @param simulationTime
 * This parameter is responsible for total length of simulation and all
 * the events with time > simulationTime will not be registered
 *
 * Note:
 * This version doesn't support @SPIKING_NN::EventType::SCHEDULED_RELAXATION,
 * will be supported later.
 *
 * Description:
 * This algorithm does processing of events stepwise. First of all it
 * selects first of available buckets and processes all the events
 * registered for that period of time chronologically.
 *
 *
 *
 * If current event is INCOMING_SPIKE, neuron potential will be updated
 * If neuron state turned to inconsistent or just the event is
 * DELAYED_ACTIVATION, new spikes for further neurons will be
 * registered.
 * If neuron state was inconsistent, this incoming spike should be processed
 * but normalization shouldn't be applied
 * If neuron is inconsistent after normalization, DELAYED_ACTIVATION will be
 * registered after its refractory period.
 */
void PreciseEventManager::RunSimulation( SPIKING_NN::Time simulationTime, bool useSTDP )
{
    for ( BucketId bucketId = 0; bucketId * SPIKING_NN::TIME_STEP < simulationTime; ++bucketId ) {
        if ( eventBuckets.find( bucketId ) == eventBuckets.end()) {
            continue;
        }
        for ( auto time_event: eventBuckets[bucketId] ) {

            const SPIKING_NN::EventKey &key = time_event.first;
            SPIKING_NN::EventValue &spike = time_event.second;

            INeuron &neuron = *key.neuronPtr;
            if ( spike.type == SPIKING_NN::EVENT_TYPE::INCOMING_SPIKE ) {
                neuron.ProcessInputSpike( key.time, spike.potential );
            }

//            TODO: Fix STDP logic here
//            if ( useSTDP ) {
//                const ISynapses &inputSynapses = neuron.GetInputSynapses();
//                for ( ISynapse *synapse: inputSynapses ) {
//                    synapse->RegisterPostSynapticSpike( event.key.time );
//                }
//                for ( ISynapse *synapse: synapses ) {
//                    synapse->RegisterPreSynapticSpike( event.key.time );
//                }
//            }

            if ( neuron.IsConsistent()) {
                /*
                 * We can get here in different ways:
                 * 1. Just not enough potential to spike
                 * 2. Overheated neuron was cooled down by inhibitory connections
                 * 3. Overheated neuron was cooled down with leakage factor
                 */
                continue;
            }

            // Register outgoing spikes for that neuron
            for ( ISynapse *synapse: neuron.GetOutputSynapses()) {
                RegisterSpikeEvent( {
                                            key.time + synapse->GetDelay(),
                                            synapse->GetPostSynapticNeuron(),
                                    },
                                    {
                                            synapse->GetStrength(),
                                            SPIKING_NN::EVENT_TYPE::INCOMING_SPIKE
                                    } );
            }

//            Next step is to normalize potential of neuron (it can be still overheated if current potential is too
//            high)
            neuron.NormalizePotential( key.time );

//            If neuron is not consistent after normalization we have plan delayed activation after refractory period
            if ( !neuron.IsConsistent()) {
                RegisterSpikeEvent( {
                                            key.time + neuron.GetTRef(),
                                            key.neuronPtr
                                    },
                                    {
                                            0,
                                            SPIKING_NN::EVENT_TYPE::DELAYED_ACTIVATION
                                    } );
            }
        }
    }
    eventBuckets.clear();
}

void PreciseEventManager::RegisterSpikeEvent( const SPIKING_NN::EventKey &key, const SPIKING_NN::EventValue &spike )
{
    BucketId bucketId = GetBucketId( key.time );
    if ( eventBuckets.find( bucketId ) == eventBuckets.end()) {
        eventBuckets[bucketId] = {{key, spike}};
    } else {
        // Single lookup update for spike value
        auto it = eventBuckets[bucketId].insert( std::make_pair( key, spike ));
        if ( !it.second ) {
            // TODO ( use regular class for EventValue to override += operator)
            it.first->second = it.first->second + spike;
        }
    }
}


PreciseEventManager::PreciseEventManager() = default;


BucketId PreciseEventManager::GetBucketId( SPIKING_NN::Time time )
{
    return static_cast<BucketId>(time / SPIKING_NN::TIME_STEP);
}

void PreciseEventManager::RegisterSample( const SPIKING_NN::Sample &sample, const SPIKING_NN::Layer &input )
{
    for ( auto timingId = 0; timingId < sample.size(); ++timingId ) {
        RegisterSpikeEvent(
                {
                        sample[timingId],
                        input[timingId]
                },
                {
                        1,
                        SPIKING_NN::EVENT_TYPE::INCOMING_SPIKE
                } );
    }
}

void PreciseEventManager::RegisterSpikeTrain( const SPIKING_NN::SpikeTrain &sample, ILayer &input )
{
    for ( int activationId = 0; activationId < sample.size(); ++activationId ) {
        for ( auto t: sample[activationId] ) {
            RegisterSpikeEvent( {
                                        t,
                                        input[activationId]
                                },
                                {
                                        1,
                                        SPIKING_NN::EVENT_TYPE::INCOMING_SPIKE
                                } );
        }
    }
}