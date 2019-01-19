#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include "LifNeuron.h"
#include "Synapse.h"
#include "EventManager.h"
#include <memory>
#include <random>
#include <time.h>

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
const int HIDDEN1_SIZE = 10;
const int HIDDEN2_SIZE = 10;
const int OUTPUT_SIZE = 3;
const float LEARNING_RATE = 0.001;



void ReadData( const std::string &path, Dataset& data, int inputSize, int simulationTime);
void createSynapses(Layer &layer1, Layer &layer2,
                    int size1, int size2);
void createLaterInibitionSynapses(Layer &layer, int size);

void RelaxLayer(Layer &layer, int size);
void RelaxOutputLayer( Layer &outputLayer, const Target &target, int size );
void RelaxInputLayer( Layer &layer);

void InitLayer( Layer &layer, int size, int nextLayerSize );


void GradStep( Layer &layer, float learningRate );
int RegisterSample( SpikeTrain &sample, EventManager &manager, Layer &input);
void ResetLayer( Layer &layer );

float SoftMaxLoss( Layer &outputLayer, const Target &target,
                   std::vector<float> &softMax, std::vector<float> &deltas, float *S);

template <class T> int Argmax( const std::vector<T> &vector );

int main() {

    Layer inputLayer;
    InitLayer( inputLayer, INPUT_SIZE, HIDDEN1_SIZE );
    Layer hidden1;
    InitLayer( hidden1, HIDDEN1_SIZE, HIDDEN2_SIZE );
    Layer hidden2;
    InitLayer( hidden2, HIDDEN2_SIZE, OUTPUT_SIZE );
    Layer output;
    InitLayer( output, OUTPUT_SIZE, 0);

    createSynapses(inputLayer, hidden1, INPUT_SIZE, HIDDEN1_SIZE);
    createSynapses(hidden1, hidden2, HIDDEN1_SIZE, HIDDEN2_SIZE);
    createSynapses(hidden2, output, HIDDEN2_SIZE, OUTPUT_SIZE);
//    createLaterInibitionSynapses(hidden1, HIDDEN1_SIZE);
//    createLaterInibitionSynapses(hidden2, HIDDEN2_SIZE);
    for ( int neuronId = 0; neuronId < OUTPUT_SIZE; neuronId++ ) {
        output[neuronId].get()->sigma_mu = 0;
    }
    for ( int neuronId = 0; neuronId < INPUT_SIZE; neuronId++ ) {
        inputLayer[neuronId].get()->sigma_mu = 0;
    }
    for ( int neuronId = 0; neuronId < HIDDEN1_SIZE; neuronId++ ) {
        hidden1[neuronId].get()->sigma_mu = 0;
    }
    for ( int neuronId = 0; neuronId < HIDDEN2_SIZE; neuronId++ ) {
        hidden2[neuronId].get()->sigma_mu = 0;
    }


    Dataset data;
    ReadData("/home/izeren/projects/snn_projects/brian2_tests/data/bindsnet/iris.data", data, INPUT_SIZE, 50);

    for ( int epochId = 0; epochId < 3000; epochId++ ) {
        float train_good_counter = 0;
        float train_activity = 0;
        float totalTrainLoss = 0;
        float totalTestLoss = 0;
        for ( int sampleId = 0; sampleId < data.xTrain.size(); sampleId++ ) {
            EventManager eventManager( 50 );
            int sampleSize = RegisterSample( data.xTrain[sampleId], eventManager, inputLayer );
            eventManager.RunSimulation();
//            std::cout << "NumSpikes for sample: " << sampleId << " " << eventManager.eventCounter;
//            std::cout << " SampleSize: " << sampleSize << " Activity: " << eventManager.eventCounter - sampleSize << "\n";
            train_activity += eventManager.eventCounter - sampleSize;
            Target target( OUTPUT_SIZE, 0 );
            target[data.yTrain[sampleId]] = 1;

            float S = 0;
            std::vector<float> softMax;
            std::vector<float> deltas(OUTPUT_SIZE, 0);
            float loss = SoftMaxLoss( output, target, softMax, deltas, &S );
            totalTrainLoss += loss;
            int res = Argmax(softMax);
//            std::cout << softMax[0] << " " << softMax[1] << " " << softMax[2] << "\n";
            if ( res == data.yTrain[sampleId] ) {
                train_good_counter += 1;
            }
//            std::cout << res << " " << data.yTrain[sampleId] << "\n";

            RelaxOutputLayer( output, target, OUTPUT_SIZE );
            RelaxLayer( hidden2, HIDDEN2_SIZE );
            RelaxLayer( hidden1, HIDDEN1_SIZE );
            RelaxInputLayer( inputLayer );

            GradStep( output, LEARNING_RATE );
            GradStep( hidden2, LEARNING_RATE );
            GradStep( hidden1, LEARNING_RATE );
            GradStep( inputLayer, LEARNING_RATE );

            ResetLayer( output );
            ResetLayer( hidden2 );
            ResetLayer( hidden1 );
            ResetLayer( inputLayer );
        }


        int tp = 0;
        int fp = 0;
        int tn = 0;
        int fn = 0;
        float test_good_counter = 0;
        for ( int sampleId = 0; sampleId < data.xTest.size(); sampleId++ ) {
            EventManager eventManager( 50 );
            RegisterSample( data.xTest[sampleId], eventManager, inputLayer );
            eventManager.RunSimulation();

            int t = data.yTest[sampleId];
            Target target( OUTPUT_SIZE, 0 );
            target[data.yTest[sampleId]] = 1;
            float S = 0;
            std::vector<float> softMax;
            std::vector<float> deltas(OUTPUT_SIZE, 0);
            float loss = SoftMaxLoss( output, target, softMax, deltas, &S );
            totalTestLoss += loss;
//
            int res = Argmax(softMax);
//            std::cout << softMax[0] << " " << softMax[1] << " " << softMax[2] << "\n";
            if ( res == t ) {
                test_good_counter += 1;
            }
//            std::cout << res << " " << t << "\n";
            ResetLayer( output );
            ResetLayer( hidden2 );
            ResetLayer( hidden1 );
            ResetLayer( inputLayer );

        }
        std::cout << train_good_counter / data.yTrain.size() << " " << test_good_counter / data.yTest.size() << " ";
        std::cout << " Average activity: " << train_activity / data.yTrain.size() << " ";
        std::cout << " Average train loss: " << totalTrainLoss / data.yTrain.size();
        std::cout << " Average test loss: " << totalTestLoss / data.yTest.size() << "\n";

    }

}

void createSynapses(Layer &layer1, Layer &layer2,
                    int size1, int size2) {
    std::default_random_engine generator;
    generator.seed( clock() );
    float limit = sqrt( 3. / (  size2 ) );
    std::uniform_real_distribution<float> distribution( -limit, limit );

    for (int prevId = 0; prevId < size1; prevId++) {
        for (int nextId = 0; nextId < size2; nextId++) {
            LifNeuron *prev = layer1[prevId].get();
            LifNeuron *next = layer2[nextId].get();
            prev->outputSynapses.emplace_back(Synapse());
            prev->outputSynapses.back().strength = distribution(generator);
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

float SoftMaxLoss( Layer &outputLayer, const Target &target,
        std::vector<float> &softMax, std::vector<float> &deltas, float *S) {
    float expSum = 0;
    float loss = 0;
    for ( auto neuron: outputLayer ) {
        softMax.push_back(exp(neuron.get()->a));
        expSum += softMax.back();
    }
    for ( int neuronId = 0; neuronId < outputLayer.size(); neuronId++ ) {
        softMax[neuronId] /= expSum;
        deltas[neuronId] = softMax[neuronId] - target[neuronId];
        *S += softMax[neuronId] * deltas[neuronId];
        loss += deltas[neuronId] * deltas[neuronId];
    }
    return loss;
}

void RelaxOutputLayer( Layer &outputLayer, const Target &target, int size ) {
    float Sa = 0;
    float S = 0;
    std::vector<float> softMax;
    std::vector<float> deltas(size, 0);
    SoftMaxLoss( outputLayer, target, softMax, deltas, &S );
    for ( int softMaxId = 0; softMaxId < size; softMaxId++ ) {
        outputLayer[softMaxId].get()->grad = softMax[softMaxId] *
                (deltas[softMaxId] - S) / outputLayer[softMaxId].get()->vMaxThresh;
        Sa += outputLayer[softMaxId].get()->a;
    }
    for ( int neuronId = 0; neuronId < size; neuronId++ ) {
        LifNeuron *n = outputLayer[neuronId].get();
        n->DlDV = n->grad * (-(1 + n->sigma_mu) * n->a / exp( 1 / n->tau ) + n->sigma_mu * Sa);
    }
}

void RelaxLayer( Layer &layer, int size ) {
    float Sa = 0;
    for ( auto neuron: layer ) {
        Sa += neuron.get()->a;
    }
    for ( auto neuron: layer ) {
        neuron.get()->Backward(Sa);
    }
}

void RelaxInputLayer( Layer &layer) {
    float Sa = 0;
    for ( auto neuron: layer ) {
        Sa += neuron.get()->a;
        neuron.get()->Backward(Sa);
        neuron.get()->DlDV = 0;
        neuron.get()->grad = 0;
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
        if ( neuron.get()->vMaxThresh < 0 ) {
            neuron.get()->vMaxThresh = 0.001;
        }
        for ( auto synapse: neuron.get()->outputSynapses ) {
            if ( synapse.updatable ) {
                synapse.strength -= synapse.DlDw * learningRate;
            }
        }
    }
}

int RegisterSample( SpikeTrain &sample, EventManager &manager, Layer &input) {
    int sampleSize = 0;
    for ( int tick = 0; tick < sample.size(); tick++ ) {
        for ( int neuronId = 0; neuronId < input.size(); neuronId++ ) {
            if ( sample[tick][neuronId] ) {
                manager.RegisterSpikeEvent( input[neuronId].get(), tick );
                sampleSize += 1;
            }
        }
    }
    return sampleSize;
}

void InitLayer( Layer &layer, int size, int nextLayerSize ) {
    float limit;
    if ( nextLayerSize ) {
        limit = 1 * sqrt(3. / nextLayerSize );
    } else {
        limit = 1;
    }

    layer = Layer(size, 0);
    for ( int neuronId=0; neuronId < size; neuronId++ ) {
        layer[neuronId] = std::make_shared<LifNeuron>();
        layer[neuronId].get()->vMaxThresh = limit;
    }
}

void ResetLayer( Layer &layer ) {
    for ( auto neuron: layer ) {
        neuron.get()->Reset();
    }
}

template <class T> int Argmax( const std::vector<T> &vector ) {
    int maxId = 0;
    for ( int index = 0; index < vector.size(); index++ ) {
        if ( vector[index] > vector[maxId]) {
            maxId = index;
        }
    }
    return maxId;
}