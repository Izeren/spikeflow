#include "ILayerBuilder.h"
#include "DenseLifLayer.h"
#include "DenseVanillaLayer.h"

ILayer *DenseLifLayerBuilder::Build( LayerMeta meta ) const
{
    return new DenseLifLayer( meta );
}

ILayer *DenseVanillaLayerBuilder::Build( LayerMeta meta ) const
{
    return new DenseVanillaLayer( meta );
}
