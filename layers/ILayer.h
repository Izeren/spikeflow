#pragma once

#include "../stats/LayerStats.h"
#include "INeuron.h"
#include "INeuronBuilder.h"
#include "ISynapseBuilder.h"
#include "LayerMeta.h"

class ILayer {
public:
    explicit ILayer( LayerMeta _meta );

    virtual ~ILayer();

    virtual void Init( size_t nextLayerSize, std::default_random_engine &generator, float z ) = 0;

    virtual ILayer &Relax( SPIKING_NN::Time time ) = 0;

    virtual ILayer &LogBasicStats() = 0;

    virtual ILayer &ResetStats() = 0;

    virtual ILayer &ResetPotentials() = 0;

    virtual ILayer &ResetGrad() = 0;

    virtual ILayer &
    GradStep( size_t batchSize, float learningRateV, float learningRateW, float BETA, bool isInput,
              float LAMBDA, bool isOutput ) = 0;

    virtual ILayer &Backward( const std::vector<float> &deltas ) = 0;

    virtual ILayer &Forward( ) = 0;

    virtual ILayer &BindWithNext( ILayer &nextLayer, const ISynapseBuilder &synapseBuilder );

    virtual std::string ToString() const = 0;

    friend std::ostream &operator<<( std::ostream &out, const ILayer &layer );

    INeuron *operator[]( size_t idx );

    const INeuron *operator[]( size_t idx ) const;

    size_t GetSize() const;

    std::string GetName() const;

    float GetZShift();

    std::vector<INeuron *> neurons;
protected:
    LayerStats stats;
    LayerMeta meta;
};