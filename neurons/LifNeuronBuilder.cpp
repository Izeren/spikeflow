#include "LifNeuronBuilder.h"
#include "LifNeuron.h"

INeuron *LifNeuronBuilder::build() const
{
    return new LifNeuron();
}
