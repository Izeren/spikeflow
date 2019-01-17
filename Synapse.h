#pragma once
#include <vector>

class LifNeuron;
class Synapse;
typedef std::vector<Synapse> SynapseList;

class Synapse {
public:
    Synapse( bool updatable, float strength, float x, LifNeuron &previous, LifNeuron &next );
    LifNeuron &GetNext() const;
    float GetStrength() const;
    void IncreaseX(float delta);

private:
    bool updatable;
    float strength;
    float x;
    LifNeuron& previous;
    LifNeuron& next;

};