#include <ISynapseBuilder.h>
#include "ILayer.h"

ILayer::ILayer( std::string _name, size_t _size, const INeuronBuilder &_neuronBuilder ) :
        name( std::move( _name )), size( _size ), neuronBuilder( _neuronBuilder )
{

    neurons.resize( _size );
    for ( auto idx = 0; idx < _size; ++idx ) {
        neurons[idx] = neuronBuilder.build();
    }
}

INeuron *ILayer::operator[]( size_t idx )
{
    return neurons[idx];
}

const INeuron *ILayer::operator[]( size_t idx ) const
{
    return neurons[idx];
}

std::ostream &operator<<( std::ostream &out, const ILayer &layer )
{
    return out << layer.ToString();
}

ILayer::~ILayer()
{
    for ( auto neuron: neurons ) {
        delete neuron;
    }
}

ILayer &ILayer::BindWithNext( ILayer &nextLayer, ISynapseBuilder &synapseBuilder )
{
    size_t size1 = size;
    size_t size2 = nextLayer.size;
    for ( int prevId = 0; prevId < size1; prevId++ ) {
        for ( int nextId = 0; nextId < size2; nextId++ ) {
            INeuron *prev = neurons[prevId];
            INeuron *next = nextLayer[nextId];
            ISynapse *synapsePtr = synapseBuilder.build( size1, size2, prev, next );
            prev->AddOutputSynapse( synapsePtr );
            next->AddInputSynapse( synapsePtr );
        }
    }
    return nextLayer;
}