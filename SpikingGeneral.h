#pragma once

#include <map>
#include <vector>

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
    const float TIME_STEP = 0.01;


    /**
     * This constant is used to check if there was no spikes on the
     * output neuron.
     */
    const float EPS = 1e-5;


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
    enum EVENT_TYPE {
        INCOMING_SPIKE,
        DELAYED_ACTIVATION,
        SCHEDULED_RELAXATION
    };


    /**
     * This enumeration is to distinguish different type of neurons on
     * insertion to the network
     */
    enum NEURON_TYPE {
        INPUT,
        OUTPUT,
        HIDDEN
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
        EVENT_TYPE type;
    } Event;

    /**
     * This is convenient mapping from time where event occurs to the all
     * parameters of event
     */
    typedef std::map<Time, Event> EventBucket;

    /**
     * Just a sample of data as vector of float numbers, size of vector
     * should be the same as the number of input neurons, not less.
     */
    typedef std::vector<float> Sample;


    /**
     * For current version it is assumed that all the tasks are simple
     * classification problems with integer class indices
     */
    typedef int Target;


    /**
     * This structure is responsible for aggregation of
     * 1. Train part of samples
     * 2. Train part of labels
     * 3. Test part of samples
     * 4. Test part of labels
     */
    typedef struct {
        std::vector<Sample> xTrain;
        std::vector<Target> yTrain;
        std::vector<Sample> xTest;
        std::vector<Target> yTest;
    } Dataset;


    /**
     * This score is used for losses and quality metrics
     */
    typedef float Score;


    /**
     * This type is to denote aggregation of all activity on the output
     * layer
     */
    typedef std::vector<float> Output;


    /**
     * This function type is common for different loss function and is
     * used for abstract scoring implementation in @INetwork
     */
    typedef Score (*LossFunction)(std::vector<Output> &predictions, std::vector<Target>& labels);


    /**
     * It is aggregation of neuron to a group in the meaning of layer
     * This is very convenient part of perceptron definition.
     */
    typedef std::vector<INeuron *> Layer;



};