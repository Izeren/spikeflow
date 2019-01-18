#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include "LifNeuron.h"
#include "Synapse.h"
#include "EventManager.h"
#include <memory>

#include <eigen3/Eigen/Dense>

using Eigen::MatrixXd;
typedef std::vector<std::shared_ptr<LifNeuron> > Layer;
typedef std::vector<float> Target;
typedef std::vector<std::vector<int > > SpikeTrain;
typedef struct {
    std::vector<SpikeTrain> xTrain;
    std::vector<float> yTrain;
    std::vector<SpikeTrain> xTest;
    std::vector<float> yTest;
} Dataset;

const int INPUT_SIZE = 4;
const int HIDDEN1_SIZE = 2;
const int HIDDEN2_SIZE = 2;
const int OUTPUT_SIZE = 2;
const float LEARNING_RATE = 0;



void ReadData( const std::string &path, Dataset& data, int inputSize, int simulationTime);
void createSynapses(Layer &layer1, Layer &layer2,
                    int size1, int size2);
void createLaterInibitionSynapses(Layer &layer, int size);

void relaxLayer(Layer &layer, int size);
void RelaxOutputLayer( Layer &outputLayer, const Target &target, int size );

void createLayer(Layer &layer, int size);


void GradStep( Layer &layer, float learningRate );
void RegisterSample( SpikeTrain &sample, EventManager &manager, Layer &input);

int main() {

    Layer inputLayer(INPUT_SIZE, std::make_shared<LifNeuron>());
    Layer hidden1(HIDDEN1_SIZE, std::make_shared<LifNeuron>());
    Layer hidden2(HIDDEN2_SIZE, std::make_shared<LifNeuron>());
    Layer output(OUTPUT_SIZE, std::make_shared<LifNeuron>());

    createSynapses(inputLayer, hidden1, INPUT_SIZE, HIDDEN1_SIZE);
    createSynapses(hidden1, hidden2, HIDDEN1_SIZE, HIDDEN2_SIZE);
    createSynapses(hidden2, output, HIDDEN2_SIZE, OUTPUT_SIZE);
    createLaterInibitionSynapses(hidden1, HIDDEN1_SIZE);
    createLaterInibitionSynapses(hidden2, HIDDEN2_SIZE);
    createLaterInibitionSynapses(output, OUTPUT_SIZE);

    Dataset data;
    ReadData("/home/izeren/projects/snn_projects/brian2_tests/data/bindsnet/xor.data", data, INPUT_SIZE, 50);


    for ( int sampleId = 0; sampleId < data.xTrain.size(); sampleId++ ) {
        EventManager eventManager(50);
        RegisterSample(data.xTrain[sampleId], eventManager, inputLayer);
        eventManager.RunSimulation();
        std::cout << "NumSpikes for sample: " << sampleId << " " << eventManager.eventCounter << "\n";

        Target target(OUTPUT_SIZE, 0);
        target[data.yTrain[sampleId]] = 1;

        RelaxOutputLayer(output, target, OUTPUT_SIZE);
        relaxLayer(hidden2, HIDDEN2_SIZE);
        relaxLayer(hidden1, HIDDEN1_SIZE);
        relaxLayer(inputLayer, INPUT_SIZE);

        GradStep(output, LEARNING_RATE);
        GradStep(hidden2, LEARNING_RATE);
        GradStep(hidden1, LEARNING_RATE);
        GradStep(inputLayer, LEARNING_RATE);

        EventManager resetManager(50);
        resetManager.RunSimulation();
    }

    int tp = 0;
    int fp = 0;
    int tn = 0;
    int fn = 0;
    int good_counter = 0;
    for ( int sampleId = 0; sampleId < data.xTest.size(); sampleId++ ) {
        EventManager eventManager(50);
        RegisterSample(data.xTest[sampleId], eventManager, inputLayer);
        eventManager.RunSimulation();

        int t = data.yTest[sampleId];
        int res = output[0].get()->a > output[1].get()->a;
        if ( res == t ) {
            good_counter += 1;
        }
        std::cout << res << " " << t << "\n";

        EventManager resetManager(50);
        resetManager.RunSimulation();
    }
    std::cout << good_counter / data.yTest.size() << " " << data.xTest.size() << "\n";


}

void createSynapses(Layer &layer1, Layer &layer2,
                    int size1, int size2) {
    for (int prevId = 0; prevId < size1; prevId++) {
        for (int nextId = 0; nextId < size2; nextId++) {
            LifNeuron *prev = layer1[prevId].get();
            LifNeuron *next = layer2[nextId].get();
            prev->outputSynapses.emplace_back(Synapse());
            prev->outputSynapses.back().previous = prev;
            prev->outputSynapses.back().next = next;
        }
    }
}

void createLaterInibitionSynapses(Layer &layer, int size) {
    for (int prevId = 0; prevId < size; prevId++) {
        for (int nextId = 0; nextId < size; nextId++) {
            if ( prevId != nextId ) {
                LifNeuron *prev = layer[prevId].get();
                LifNeuron *next = layer[nextId].get();
                prev->outputSynapses.emplace_back(Synapse());
                prev->outputSynapses.back().strength = -1;
                prev->outputSynapses.back().updatable = false;
                prev->outputSynapses.back().previous = prev;
                prev->outputSynapses.back().next = next;
            }
        }
    }
}

void RelaxOutputLayer( Layer &outputLayer, const Target &target, int size ) {
    float Sa = 0;
    for ( int neuronId = 0; neuronId < size; neuronId++ ) {
        outputLayer[neuronId].get()->grad = 2 * (outputLayer[neuronId].get()->a - target[neuronId]);
        Sa += outputLayer[neuronId].get()->a;
    }
    for ( int neuronId = 0; neuronId < size; neuronId++ ) {
        LifNeuron *n = outputLayer[neuronId].get();
        n->DlDV = n->grad * (-(1 + n->sigma_mu) * n->a + n->sigma_mu * Sa) / n->vMaxThresh;
    }
}

void relaxLayer( Layer &layer, int size ) {
    float Sa = 0;
    for ( auto neuron: layer ) {
        Sa += neuron.get()->a;
        neuron.get()->Backward(Sa);
    }
}

void ReadData( const std::string &path, Dataset& data, int inputSize, int simulationTime) {
    std::ifstream in(path);
    int n;
    in >> n;
    data.xTrain = std::vector<SpikeTrain>(n, SpikeTrain(simulationTime, std::vector<int>(inputSize)));
    data.yTrain = std::vector<float>(n);
    for ( int sampleId = 0; sampleId < n; sampleId++ ) {
        for ( int timeId = 0; timeId < simulationTime; timeId++ ) {
            for ( int neuronId = 0; neuronId < inputSize; neuronId++ ) {
                in >> data.xTrain[sampleId][timeId][neuronId];
            }
        }
    }
    for ( int sampleId = 0; sampleId < n; sampleId++ ) {
        in >> data.yTrain[sampleId];
    }
    in >> n;
    data.xTest = std::vector<SpikeTrain>(n, SpikeTrain(simulationTime, std::vector<int>(inputSize)));
    data.yTest= std::vector<float>(n);
    for ( int sampleId = 0; sampleId < n; sampleId++ ) {
        for ( int timeId = 0; timeId < simulationTime; timeId++ ) {
            for ( int neuronId = 0; neuronId < inputSize; neuronId++ ) {
                in >> data.xTest[sampleId][timeId][neuronId];
            }
        }
    }
    for ( int sampleId = 0; sampleId < n; sampleId++ ) {
        in >> data.yTest[sampleId];
    }
}

void GradStep( Layer &layer, float learningRate ) {
    for ( auto neuron: layer ) {
        neuron.get()->vMaxThresh -= neuron.get()->DlDV * learningRate;
        for ( auto synapse: neuron.get()->outputSynapses ) {
            if ( synapse.updatable ) {
                synapse.strength -= synapse.DlDw * learningRate;
            }
        }
    }
}

void RegisterSample( SpikeTrain &sample, EventManager &manager, Layer &input) {
    for ( int tick = 0; tick < sample.size(); tick++ ) {
        for ( int neuronId = 0; neuronId < input.size(); neuronId++ ) {
            if ( sample[tick][neuronId] ) {
                manager.RegisterSpikeEvent( input[neuronId].get(), tick );
            }
        }
    }
}