#pragma once

#include "IEventManager.h"
#include "../layers/ILayer.h"
#include <queue>


/**
 * This type is used to denote unique number of bucket on timeline
 * based on integer part of time by the module of @SPIKING_NN::TIME_STEP
 */
typedef size_t BucketId;


/**
 * This class is responsible for clever time processing.
 * Using this class one is now able to use float synaptic delays
 * It is very important for time based encodings of SNN.
 * You need to remember that this class works under several restrictions.
 *
 * 1. Because delays are float this class doesn't check the collisions of
 * events by the time. It is supposed that all spiking events are separated
 * in the full timeline. So it is now important not to use discrete
 * delays because of multiple collisions and spiking drops.
 *
 * 2. You can't use synaptic delays less than @SPIKING_NN::TIME_STEP
 * because it is very expensive to do double checks of eventBucket.
 * Current algorithm assumes all new spikes shouldn't occur at the current
 * bucket under calculations.
 *
 * 3. Refractory period of neuron should also be greater than
 * @SPIKING_NN::TIME_STEP, this is crucial, because we can get overheated
 * neurons which will spike with tRefractory step and those spikes should
 * be placed to the further EventBuckets.
 *
 * 4. Because of reasons above this version doesn't support easy relaxation
 * of output neurons at the stage of event registration and also stepwise
 * relaxations. Will be updated later.
 *
 * So the concept of eventBucket is follows. Event bucket is ordered
 * map of Events. The key is the time of event. So all the spikes are
 * processed by the time. If there are several events with the same time,
 * only the last one will be saved. All other events will be overwritten.
 * So it is called bucket because it aggregates all the events with time
 * in [t1, t2], where t2 - t1 == @SPIKING_NN::TIME_STEP and the whole
 * timeline is splitted into those buckets.
 */
class PreciseEventManager : public IEventManager {
public:
    void RegisterSpikeEvent( const SPIKING_NN::EventKey &key, const SPIKING_NN::EventValue &spike );

    void RunSimulation( SPIKING_NN::Time time, bool useSTDP ) override;

    void RegisterSample( const SPIKING_NN::Sample &sample, const SPIKING_NN::Layer &input ) override;

    void RegisterSpikeTrain( const SPIKING_NN::SpikeTrain &sample, ILayer &input ) override;

    PreciseEventManager();

private:
    std::map<BucketId, SPIKING_NN::EventBucket> eventBuckets;

    static BucketId GetBucketId( SPIKING_NN::Time time );

};