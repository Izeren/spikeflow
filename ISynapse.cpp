//
// Created by izeren on 3/22/19.
//

#include "ISynapse.h"
#include "INeuron.h"

float ISynapse::GetDelay() {
    return delay;
}

float ISynapse::GetStrength() {
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

ISynapse::ISynapse( float _strength, float _delay ) :
        strength( _strength ), delay( _delay ) {}

ISynapse::ISynapse( float _strength, float _delay, INeuron *_preSynapticNeuron, INeuron *_postSynapticNeuron ) :
        strength( _strength ), delay( _delay ), preSynapticNeuron( _preSynapticNeuron ),
        postSynapticNeuron( _postSynapticNeuron ) {}
