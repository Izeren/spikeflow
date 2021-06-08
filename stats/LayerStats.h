#pragma once

#include "Stat.h"

//    Synapse stats applicable to output synapses of neurons of the network
class LayerStats {
public:

    LayerStats();

    void Reset();

    Stat<float> synapseWeight;

    Stat<float> neuronMPMax;

    Stat<float> neuronMP;

    Stat<float> neuronMPInduced;

    Stat<int> inducedSpikes;

    Stat<int> spikes;

    Stat<float> gradV;

    Stat<float> gradW;

    friend std::ostream &operator<<( std::ostream &out, const LayerStats &stats );

};
