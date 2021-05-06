#pragma once

#include "SpikingGeneral.h"


class IEventManager {

public:
    IEventManager() = default;

    virtual ~IEventManager() = default;

    virtual void RunSimulation( SPIKING_NN::Time time, bool useSTDP ) = 0;

    virtual void RegisterSample( const SPIKING_NN::Sample &sample, const SPIKING_NN::Layer &input ) = 0;

};
