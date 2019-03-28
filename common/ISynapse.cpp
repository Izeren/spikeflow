//
// Created by izeren on 3/22/19.
//

#include "ISynapse.h"
#include "INeuron.h"

SPIKING_NN::Time ISynapse::GetDelay() {
    return delay;
}

SPIKING_NN::Strength ISynapse::GetStrength() {
    return strength;
}

ISynapse::~ISynapse() {
    if ( preSynapticNeuron ) {
        preSynapticNeuron->ForgetOutputSynapse( this );
    }
    if ( postSynapticNeuron ) {
        postSynapticNeuron->ForgetInputSynapse( this );
    }
}

ISynapse::ISynapse( SPIKING_NN::Strength _strength, SPIKING_NN::Time _delay ) :
        strength( _strength ), delay( _delay ) {}

ISynapse::ISynapse( SPIKING_NN::Strength _strength, SPIKING_NN::Time _delay, INeuron *_preSynapticNeuron,
                    INeuron *_postSynapticNeuron ) :
        strength( _strength ), delay( _delay ), preSynapticNeuron( _preSynapticNeuron ),
        postSynapticNeuron( _postSynapticNeuron ) {}

INeuron *ISynapse::GetPreSynapticNeuron() const {
    return preSynapticNeuron;
}

void ISynapse::SetPreSynapticNeuron( INeuron *preSynapticNeuron ) {
    ISynapse::preSynapticNeuron = preSynapticNeuron;
}

INeuron *ISynapse::GetPostSynapticNeuron() const {
    return postSynapticNeuron;
}

void ISynapse::SetPostSynapticNeuron( INeuron *postSynapticNeuron ) {
    ISynapse::postSynapticNeuron = postSynapticNeuron;
}

void ISynapse::SetStrength( SPIKING_NN::Strength strength ) {
    ISynapse::strength = strength;
}

void ISynapse::SetDelay( SPIKING_NN::Time delay ) {
    ISynapse::delay = delay;
}
