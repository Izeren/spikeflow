//
// Created by izeren on 3/22/19.
//

#include "INeuron.h"
#include "ISynapse.h"

const ISynapses &INeuron::GetInputSynapses() const {
    return inputSynapses;
}

const ISynapses &INeuron::GetOutputSynapses() const {
    return outputSynapses;
}

void INeuron::AddInputSynapse( ISynapse *synapse ) {
    inputSynapses.insert( synapse );
}

void INeuron::AddOutputSynapse( ISynapse *synapse ) {
    outputSynapses.insert( synapse );
}

INeuron::~INeuron() {
    for ( auto synapse: inputSynapses ) {
        delete synapse;
    }
    for ( auto synapse: outputSynapses ) {
        delete synapse;
    }
}

void INeuron::ForgetOutputSynapse( ISynapse *synapse ) {
    outputSynapses.erase( synapse );
}

void INeuron::ForgetInputSynapse( ISynapse *synapse ) {
    inputSynapses.erase( synapse );
}

INeuron::INeuron( SPIKING_NN::Potential _potential, SPIKING_NN::Time _tRef, bool _isConsistent ) :
        potential( _potential ), tRef( _tRef ), consistent( _isConsistent ) {}

SPIKING_NN::Time INeuron::GetTRef() const {
    return tRef;
}

void INeuron::SetTRef( SPIKING_NN::Time tRef ) {
    INeuron::tRef = tRef;
}

bool INeuron::IsConsistent() const {
    return consistent;
}

void INeuron::SetConsistent( bool consistent ) {
    INeuron::consistent = consistent;
}

