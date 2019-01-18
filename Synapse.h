#pragma once
#include <vector>

class LifNeuron;
class Synapse;
typedef std::vector<Synapse> SynapseList;

class Synapse {
public:
    Synapse( bool updatable, float strength, float x, LifNeuron *previous, LifNeuron *next );
    Synapse();

    bool updatable;
    float strength;
    float x;
    float DaDx;
    float DlDw;
    LifNeuron *previous;
    LifNeuron *next;

};