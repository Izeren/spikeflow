#include <cmath>

#include "ISynapse.h"
#include "INeuron.h"

const SPIKING_NN::Strength ISynapse::outputSpikeStrength = 1;
const SPIKING_NN::Strength ISynapse::inputSpikeStrength = 1;

const SPIKING_NN::Time ISynapse::tauOutput = SPIKING_NN::TIME_STEP * 10;
const SPIKING_NN::Time ISynapse::tauInput = SPIKING_NN::TIME_STEP * 10;

const SPIKING_NN::Time ISynapse::DEFAULT_SYNAPSE_DELAY = 1.f;
const SPIKING_NN::Strength ISynapse::DEFAULT_SYNAPSE_STRENGTH = 1.f;
const SPIKING_NN::Strength ISynapse::DEFAULT_LATERAL_SYNAPSE_STRENGTH = -1.f;
const bool ISynapse::DEFAULT_SYNAPSE_UPDATABILITY = true;


SPIKING_NN::Time ISynapse::GetDelay()
{
    return delay;
}

SPIKING_NN::Strength ISynapse::GetStrength()
{
    return strength;
}

ISynapse::~ISynapse()
{
    if ( preSynapticNeuron ) {
        preSynapticNeuron->ForgetOutputSynapse( this );
    }
    if ( postSynapticNeuron ) {
        postSynapticNeuron->ForgetInputSynapse( this );
    }
}

ISynapse::ISynapse( bool _isUpdatable, SPIKING_NN::Strength _strength, SPIKING_NN::Time _delay ) :
        updatable( _isUpdatable ), strength( _strength ), delay( _delay ), inputRelaxation( 0 ),
        outputRelaxation( 0 ) { }

ISynapse::ISynapse(
        bool _isUpdatable,
        SPIKING_NN::Strength _strength,
        SPIKING_NN::Time _delay,
        INeuron *_preSynapticNeuron,
        INeuron *_postSynapticNeuron
) : updatable( _isUpdatable ), strength( _strength ),
    delay( _delay ), preSynapticNeuron( _preSynapticNeuron ),
    postSynapticNeuron( _postSynapticNeuron ), inputRelaxation( 0 ), outputRelaxation( 0 ) { }

INeuron *ISynapse::GetPreSynapticNeuron() const
{
    return preSynapticNeuron;
}

void ISynapse::SetPreSynapticNeuron( INeuron *preSynapticNeuron )
{
    ISynapse::preSynapticNeuron = preSynapticNeuron;
}

INeuron *ISynapse::GetPostSynapticNeuron() const
{
    return postSynapticNeuron;
}

void ISynapse::SetPostSynapticNeuron( INeuron *postSynapticNeuron )
{
    ISynapse::postSynapticNeuron = postSynapticNeuron;
}

void ISynapse::SetStrength( SPIKING_NN::Strength strength )
{
    ISynapse::strength = strength;
}

void ISynapse::SetDelay( SPIKING_NN::Time delay )
{
    ISynapse::delay = delay;
}

void ISynapse::RegisterPreSynapticSpike( SPIKING_NN::Time time )
{
    outputTrace = outputTrace * exp(( outputRelaxation - time ) / ISynapse::tauOutput ) + ISynapse::outputSpikeStrength;
    outputRelaxation = time;
    strength += ISynapse::GetPreSynapticUpdateStrength( strength ) * outputTrace;
}

void ISynapse::RegisterPostSynapticSpike( SPIKING_NN::Time time )
{
    inputTrace = inputTrace * exp(( inputRelaxation - time ) / ISynapse::tauInput ) + ISynapse::inputSpikeStrength;
    inputRelaxation = time;
    strength -= ISynapse::GetPostSynapticUpdateStrength( strength ) * inputTrace;
}

SPIKING_NN::Strength ISynapse::GetPreSynapticUpdateStrength( SPIKING_NN::Strength strength )
{
    return 1.0;
}

SPIKING_NN::Strength ISynapse::GetPostSynapticUpdateStrength( SPIKING_NN::Strength strength )
{
    return 1.0;
}

bool ISynapse::IsUpdatable()
{
    return updatable;
}