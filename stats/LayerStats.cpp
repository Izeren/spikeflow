#include "LayerStats.h"
#include "Stat.h"

void LayerStats::Reset()
{
    synapseWeight.Reset();
    neuronMP.Reset();
    spikes.Reset();
    gradV.Reset();
    gradW.Reset();
}


LayerStats::LayerStats()
{
    Reset();
}

std::ostream &operator<<( std::ostream &out, const LayerStats &stats )
{
    char shift = '\t';
    out << "{\n";
    out << shift << "SynapseStats: " << stats.synapseWeight << "\n";
    out << shift << "NeuronMPStats: " << stats.neuronMP << "\n";
    out << shift << "SpikesStats: " << stats.spikes << "\n";
    out << shift << "GradVStats: " << stats.gradV << "\n";
    out << shift << "GradWStats: " << stats.gradW << "\n";
    out << "}\n";
    return out;
}
