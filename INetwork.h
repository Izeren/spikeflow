//
// Created by izeren on 3/22/19.
//
#pragma once
#include <vector>

class INeuron;
class ISynapse;

typedef std::vector<INeuron *> Layer;

class INetwork {

protected:

    Layer input;
    Layer output;

    /**
     * It is important note that hidden neurons not binded strictly to layers
     * It still easy to separate neurons by hidden layers if we know their sizes
     * and store all the neurons subsequently
     * This assumption gives us a huge power of indeterminate architectures
     */
    Layer hidden;
    bool isPerceptron;
    std::vector<size_t> hiddenLayersSizes;

};
