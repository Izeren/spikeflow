#pragma once

#include "ILayer.h"

class ILayerBuilder {
public:
    virtual ILayer *Build( LayerMeta meta ) const = 0;
};

class DenseLifLayerBuilder : public ILayerBuilder {
public:
    ILayer *Build( LayerMeta meta ) const override;
};

class DenseVanillaLayerBuilder : public ILayerBuilder {
public:
    ILayer *Build( LayerMeta meta ) const override;
};