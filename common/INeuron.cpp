#include "INeuron.h"
#include "ISynapse.h"

const ISynapses &INeuron::GetInputSynapses() const
{
    return inputSynapses;
}

const ISynapses &INeuron::GetOutputSynapses() const
{
    return outputSynapses;
}

void INeuron::AddInputSynapse( ISynapse *synapse )
{
    inputSynapses.insert( synapse );
}

void INeuron::AddOutputSynapse( ISynapse *synapse )
{
    outputSynapses.insert( synapse );
}

INeuron::~INeuron()
{
    while ( !inputSynapses.empty()) {
        delete *inputSynapses.begin();
    }
    while ( !outputSynapses.empty()) {
        delete *outputSynapses.begin();
    }
}

void INeuron::ForgetOutputSynapse( ISynapse *synapse )
{
    outputSynapses.erase( synapse );
}

void INeuron::ForgetInputSynapse( ISynapse *synapse )
{
    inputSynapses.erase( synapse );
}

INeuron::INeuron( SPIKING_NN::Potential _potential, SPIKING_NN::Time _tRef, bool _isConsistent ) :
        potential( _potential ), tRef( _tRef ), consistent( _isConsistent ) { }

SPIKING_NN::Time INeuron::GetTRef() const
{
    return tRef;
}

void INeuron::SetTRef( SPIKING_NN::Time tRef )
{
    INeuron::tRef = tRef;
}

bool INeuron::IsConsistent() const
{
    return consistent;
}

void INeuron::SetConsistent( bool consistent )
{
    INeuron::consistent = consistent;
}

int INeuron::GetInputSpikeCounter() const
{
    return inputSpikeCounter;
}

int INeuron::GetOutputSpikeCounter() const
{
    return outputSpikeCounter;
}

bool INeuron::operator<( const INeuron &other ) const
{
    return GetOutput() < other.GetOutput();
}

SPIKING_NN::Potential INeuron::GetInduced() const
{
    return induced;
}

void INeuron::SetInduced( float induced )
{
    INeuron::induced = induced;
}

SPIKING_NN::Potential INeuron::GetPotential() const
{
    return potential;
}

void INeuron::SetPotential( float potential )
{
    INeuron::potential = potential;
}

int INeuron::GetInductionAffectedSpikes() const
{
    return inductionAffectedSpikes;
}
