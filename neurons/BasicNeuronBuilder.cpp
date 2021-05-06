#include "BasicNeuronBuilder.h"
#include "BasicNeuron.h"

INeuron *BasicNeuronBuilder::build() const
{
    return new BasicNeuron();
}
