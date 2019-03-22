//
// Created by izeren on 3/22/19.
//

#include "INeuron.h"
#include "ISynapse.h"

INeuron::INeuron( float _potential ) : potential( _potential ) {}

const std::unordered_set<ISynapse *> &INeuron::GetInputSynapses() const {
    return inputSynapses;
}

const std::unordered_set<ISynapse *> &INeuron::GetOutputSynapses() const {
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
    outputSynapses.erase(synapse);
}

void INeuron::ForgetInputSynapse( ISynapse *synapse ) {
    inputSynapses.erase(synapse);
}
