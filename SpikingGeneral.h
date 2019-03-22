#pragma once

#include <map>

class INeuron;

class ISynapse;

namespace SPIKING_NN {


    /**
     * This is global constant responsible for time discretization
     * This discretization is used for wide spikes calculation
     * (for models like SRM) and also this discretization is used
     * for regular potential updates of neurons. The last is important
     * to calculate continuous leackages of neuron potential.
     */
    const float TIME_STEP = 0.5;


    /**
     * This potential is used in 2 different ways
     * 1. To denote the current potential of the neuron
     * 2. To denote the potential of spike, so it is
     * current influence of spike to the neuron and
     * actually is the delta of potential
     */
    typedef float Potential;


    /**
     * This type is primarily used to denote the
     * exact time of event. But also is used to denote
     * the length of time segment, so it is just the measure of time
     */
    typedef float Time;


    /**
     * This type is for potential updates and represents the strength of
     * synaptic link
     */
    typedef float Strength;


    /**
     * This enumeration is to distinguish possible behaviours of neuron
     *
     * 1. INCOMING_SPIKE: means that neuron should immediately change it
     * potential according to the activation rules.
     * 2. DELAYED_ACTIVATION: means that neuron was overheated and will
     * produce more spikes than one. First of that spikes was processed
     * during INCOMING_SPIKE
     * 3. SCHEDULED_RELAXATION: is for correct leackages processing for
     * some models we should update the potential each step of simulation
     * that's why it is important. (Also it is useful for output neuron
     * relaxation in sparse models)
     */
    enum EventType {
        INCOMING_SPIKE,
        DELAYED_ACTIVATION,
        SCHEDULED_RELAXATION
    };


    /**
     * This structure is responsible for aggregation of
     * 1. Exact time when spike should affect the neuron
     * 2. The neuron was affected by this spike.
     * 3. The delta of potential which should be applied to the neuron
     * 4. The type of event which is described in @EventType
     */
    typedef struct {
        Time time;
        INeuron *neuronPtr;
        Potential potential;
        EventType type;
    } Event;
    typedef std::map<Time, Event> EventBucket;

};