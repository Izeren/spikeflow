#include "PreciseEventManager.h"
#include "INeuron.h"
#include "ISynapse.h"

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
void PreciseEventManager::RunSimulation( SPIKING_NN::Time simulationTime ) {
    for ( BucketId bucketId = 0; bucketId * SPIKING_NN::TIME_STEP < simulationTime; ++bucketId ) {
        if ( eventBuckets.find( bucketId ) == eventBuckets.end()) {
            continue;
        }
        for ( auto time_event: eventBuckets[bucketId] ) {

            SPIKING_NN::Event &event = time_event.second;
            if ( event.type != SPIKING_NN::EVENT_TYPE::SCHEDULED_RELAXATION ) {
                spikeCounter += 1;
            }

            INeuron &neuron = *event.neuronPtr;
            bool wasConsistent = neuron.IsConsistent();
            if ( event.type == SPIKING_NN::EVENT_TYPE::INCOMING_SPIKE ) {
                neuron.ProcessInputSpike( event.time, event.potential );
            }

            if ((wasConsistent && !neuron.IsConsistent()) ||
                (event.type == SPIKING_NN::EVENT_TYPE::DELAYED_ACTIVATION)) {
                const ISynapses &synapses = neuron.GetOutputSynapses();
                for ( ISynapse *synapse: synapses ) {
                    RegisterSpikeEvent( {
                                                event.time + synapse->GetDelay(),
                                                synapse->GetPostSynapticNeuron(),
                                                synapse->GetStrength(),
                                                SPIKING_NN::EVENT_TYPE::INCOMING_SPIKE
                                        } );
                }
            }
            if ( wasConsistent && !neuron.IsConsistent()) {
                neuron.NormalizePotential( event.time );
                if ( !neuron.IsConsistent()) {
                    RegisterSpikeEvent( {
                                                event.time + neuron.GetTRef(),
                                                event.neuronPtr,
                                                0,
                                                SPIKING_NN::EVENT_TYPE::DELAYED_ACTIVATION
                                        } );
                }
            }
        }
    }
    eventBuckets.clear();
}

void PreciseEventManager::RegisterSpikeEvent( const SPIKING_NN::Event &event ) {
    BucketId bucketId = GetBucketId( event.time );
    if ( eventBuckets.find( bucketId ) == eventBuckets.end()) {
        eventBuckets[bucketId] = {{event.time, event}};
    } else {
        eventBuckets[bucketId][event.time] = event;
    }
}


PreciseEventManager::PreciseEventManager() :
        spikeCounter( 0 ) {}


BucketId PreciseEventManager::GetBucketId( SPIKING_NN::Time time ) {
    return static_cast<BucketId>(time / SPIKING_NN::TIME_STEP);
}


void PreciseEventManager::RegisterSample( const SPIKING_NN::Sample &sample, const SPIKING_NN::Layer &input ) {
    for ( auto timingId = 0; timingId < sample.size(); ++timingId ) {
        RegisterSpikeEvent( {sample[timingId], input[timingId], 0, SPIKING_NN::EVENT_TYPE::DELAYED_ACTIVATION} );
    }
}

size_t PreciseEventManager::GetSpikeCounter() const {
    return spikeCounter;
}
