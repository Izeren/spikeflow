#include <ISynapseBuilder.h>
#include "ILayer.h"

ILayer::ILayer( LayerMeta _meta ) : meta( std::move(_meta) )
{
    neurons.resize( meta.size );
    for ( auto idx = 0; idx < meta.size; ++idx ) {
        neurons[idx] = meta.neuronBuilder.build();
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

ILayer &ILayer::BindWithNext( ILayer &nextLayer, const ISynapseBuilder &synapseBuilder )
{
    size_t size1 = GetSize();
    size_t size2 = nextLayer.GetSize();
    for ( int prevId = 0; prevId < size1; prevId++ ) {
        for ( int nextId = 0; nextId < size2; nextId++ ) {
            INeuron *prev = neurons[prevId];
            INeuron *next = nextLayer[nextId];
            ISynapse *synapsePtr = synapseBuilder.Build( size1, size2, prev, next );
            prev->AddOutputSynapse( synapsePtr );
            next->AddInputSynapse( synapsePtr );
        }
    }
    return *this;
}

std::string ILayer::GetName() const
{
    return meta.name;
}

size_t ILayer::GetSize() const
{
    return meta.size;
}

float ILayer::GetZShift()
{
    return meta.zShift;
}
