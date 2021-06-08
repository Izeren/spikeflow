#include "LayerStats.h"
#include "Stat.h"

void LayerStats::Reset()
{
    synapseWeight.Reset();
    neuronMPInduced.Reset();
    neuronMPMax.Reset();
    neuronMP.Reset();
    spikes.Reset();
    inducedSpikes.Reset();
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
    out << shift << "NeuronMPThresh: " << stats.neuronMPMax << "\n";
    out << shift << "NeuronMP: " << stats.neuronMP << "\n";
    out << shift << "InducedPotential: " << stats.neuronMPInduced << "\n";
    out << shift << "InducedSpikes: " << stats.inducedSpikes << "\n";
    out << shift << "SpikesStats: " << stats.spikes << "\n";
    out << shift << "GradVStats: " << stats.gradV << "\n";
    out << shift << "GradWStats: " << stats.gradW << "\n";
    out << "}\n";
    return out;
}
