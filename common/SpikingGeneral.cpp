#include "SpikingGeneral.h"

bool SPIKING_NN::operator<( const SPIKING_NN::EventKey &key1, const SPIKING_NN::EventKey &key2 )
{
    return key1.time < key2.time || key1.time == key2.time && key1.neuronPtr < key2.neuronPtr;
}

std::ostream &operator<<( std::ostream &out, const SPIKING_NN::EventValue &event )
{
    int defPrec = 4;
    int defWidth = defPrec + 3;
    out << std::fixed << std::setprecision( defPrec ) << std::setw( defWidth ) << std::setfill( ' ' );
    out << "potential: " << event.potential << " type: " << event.type << "\n";
    return out;
}

SPIKING_NN::EventValue operator+( const SPIKING_NN::EventValue &left, const SPIKING_NN::EventValue &right )
{
    return {
            left.potential + right.potential,
            std::min( left.type, right.type )
    };
}
