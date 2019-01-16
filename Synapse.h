#pragma once

class Neuron;

class Synapse {
public:
    Synapse( bool updatable, float strength, const Neuron &previous, const Neuron &next );

private:
    bool updatable;
    float strength;
    const Neuron& previous;
    const Neuron& next;
};