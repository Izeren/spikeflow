#pragma once

#include "../stats/LayerStats.h"
#include "INeuron.h"
#include "INeuronBuilder.h"
#include "ISynapseBuilder.h"

class ILayer {
public:
    ILayer( std::string _name, size_t _size, const INeuronBuilder &_neuronBuilder );

    ~ILayer();

    virtual void Init( float alpha, size_t nextLayerSize ) = 0;

    virtual ILayer &Relax( SPIKING_NN::Time time ) = 0;

    virtual ILayer &LogBasicStats() = 0;

    virtual ILayer &ResetStats() = 0;

    virtual ILayer &ResetPotentials() = 0;

    virtual ILayer &ResetGrad() = 0;

    virtual ILayer &
    GradStep( size_t batchSize, float learningRateV, float learningRateW, float BETA, bool isInput ) = 0;

    virtual ILayer &Backward( const std::vector<float> &deltas ) = 0;

    // Returns reference to the NEXT layer (for chain binding)
    virtual ILayer &BindWithNext( ILayer &nextLayer, ISynapseBuilder &synapseBuilder );

    virtual std::string ToString() const = 0;

    friend std::ostream &operator<<( std::ostream &out, const ILayer &layer );

    size_t size;
    std::string name;
    LayerStats stats;
    const INeuronBuilder &neuronBuilder;
    std::vector<INeuron *> neurons;

    INeuron *operator[]( size_t idx );

    const INeuron *operator[]( size_t idx ) const;


};