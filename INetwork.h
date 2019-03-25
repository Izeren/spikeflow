//
// Created by izeren on 3/22/19.
//
#pragma once

#include <vector>
#include "SpikingGeneral.h"

class INeuron;

class ISynapse;

class IEventManager;

class INetwork {
public:
    virtual ~INetwork();

    virtual void InitInput( size_t size ) = 0;

    virtual void InitOutput( size_t size ) = 0;

    virtual void InitHidden( size_t size ) = 0;

    virtual void forward( const SPIKING_NN::Sample &sample, std::vector<float> &output, SPIKING_NN::Time time );

    virtual SPIKING_NN::Score
    ScoreModel( SPIKING_NN::Dataset &data, SPIKING_NN::LossFunction function, bool onTest, SPIKING_NN::Time time );


protected:

    SPIKING_NN::Layer input;
    SPIKING_NN::Layer output;

    /**
     * It is important note that hidden neurons not binded strictly to layers
     * It still easy to separate neurons by hidden layers if we know their sizes
     * and store all the neurons subsequently
     * This assumption gives us a huge power of indeterminate architectures
     */
    SPIKING_NN::Layer hidden;

    IEventManager *eventManager;

};
