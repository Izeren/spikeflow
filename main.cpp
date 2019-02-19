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
#include "MnistUtils.hpp"

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

typedef struct {
    std::vector<float> expectedOutput;
} Activity;

const int INPUT_SIZE = 2;
const int HIDDEN1_SIZE = 2;
const int HIDDEN2_SIZE = 1;
const int OUTPUT_SIZE = 2;
const float LEARNING_RATE_W = 0.0001;
const float LEARNING_RATE_V = 0.0;
const float BETA = 10;
const float LAMBDA = 0.008;





void ReadData( const std::string &path, Dataset& data, int inputSize, int simulationTime);
void createSynapses(Layer &layer1, Layer &layer2,
                    int size1, int size2);
void createLaterInibitionSynapses(Layer &layer, int size);

void RelaxLayer(Layer &layer, int size);
void RelaxOutputLayer( Layer &outputLayer, const Target &target, int size );
void RelaxInputLayer( Layer &layer);

void InitLayer( Layer &layer, int size, int nextLayerSize );


void GradStep( Layer &layer, float learningRateW, float learningRateV );
int RegisterSample( SpikeTrain &sample, EventManager &manager, Layer &input);
void ResetLayer( Layer &layer );

float SoftMaxLoss( Layer &outputLayer, const Target &target,
                   std::vector<float> &softMax, std::vector<float> &deltas, float *S);

template <class T> int Argmax( const std::vector<T> &vector );

void GenerateTestData(Dataset &data, Activity &targetActivity);
void GenerateBackPropTestData( Dataset &data, Activity &targetActivity );
void TestForward();
void TestBackProp();

void GeneratePoissonSeries( std::vector<int> &data, int averageNumSpikes, int simulationTime=0 );

template <class T> void PrintVector( const std::vector<T> &vector );


int main() {
//    TestForward();
//    TestBackProp();

    auto test = std::vector<int>(50);
    PrintVector(test);
    GeneratePoissonSeries(test, 10);
    PrintVector(test);
    MNIST::Dataset mnist;
    MNIST::ReadMnist("/home/izeren/CLionProjects/SpikeProp/mnist_data", mnist);
    std::cout << mnist.xTrain.size() << "\n";
    PrintVector(mnist.xTrain[0]);
    PrintVector(mnist.yTrain);

//    Layer inputLayer;
//    InitLayer( inputLayer, INPUT_SIZE, HIDDEN1_SIZE );
//    Layer hidden1;
//    InitLayer( hidden1, HIDDEN1_SIZE, HIDDEN2_SIZE );
//    Layer hidden2;
//    InitLayer( hidden2, HIDDEN2_SIZE, OUTPUT_SIZE );
//    Layer output;
//    InitLayer( output, OUTPUT_SIZE, 0);
//
//    createSynapses(inputLayer, hidden1, INPUT_SIZE, HIDDEN1_SIZE);
//    createSynapses(hidden1, hidden2, HIDDEN1_SIZE, HIDDEN2_SIZE);
//    createSynapses(hidden2, output, HIDDEN2_SIZE, OUTPUT_SIZE);
////    createLaterInibitionSynapses(hidden1, HIDDEN1_SIZE);
////    createLaterInibitionSynapses(hidden2, HIDDEN2_SIZE);
//    for ( int neuronId = 0; neuronId < OUTPUT_SIZE; neuronId++ ) {
//        output[neuronId].get()->sigma_mu = 0;
//    }
//    for ( int neuronId = 0; neuronId < INPUT_SIZE; neuronId++ ) {
//        inputLayer[neuronId].get()->sigma_mu = 0;
//    }
//    for ( int neuronId = 0; neuronId < HIDDEN1_SIZE; neuronId++ ) {
//        hidden1[neuronId].get()->sigma_mu = 0;
//    }
//    for ( int neuronId = 0; neuronId < HIDDEN2_SIZE; neuronId++ ) {
//        hidden2[neuronId].get()->sigma_mu = 0;
//    }
//
//
//    Dataset data;
//    ReadData("/home/izeren/projects/snn_projects/brian2_tests/data/bindsnet/xor.data", data, INPUT_SIZE, 50);
//
//    for ( int epochId = 0; epochId < 3000; epochId++ ) {
//        float train_good_counter = 0;
//        float train_activity = 0;
//        float totalTrainLoss = 0;
//        float totalTestLoss = 0;
//        for ( int sampleId = 0; sampleId < data.xTrain.size(); sampleId++ ) {
//            EventManager eventManager( 50 );
//            int sampleSize = RegisterSample( data.xTrain[sampleId], eventManager, inputLayer );
//            eventManager.RunSimulation();
////            std::cout << "NumSpikes for sample: " << sampleId << " " << eventManager.eventCounter;
////            std::cout << " SampleSize: " << sampleSize << " Activity: " << eventManager.eventCounter - sampleSize << "\n";
//            train_activity += eventManager.eventCounter - sampleSize;
//            Target target( OUTPUT_SIZE, 0 );
//            target[data.yTrain[sampleId]] = 1;
//
//            float S = 0;
//            std::vector<float> softMax;
//            std::vector<float> deltas(OUTPUT_SIZE, 0);
//            float loss = SoftMaxLoss( output, target, softMax, deltas, &S );
//            totalTrainLoss += loss;
//            int res = Argmax(softMax);
////            std::cout << softMax[0] << " " << softMax[1] << " " << softMax[2] << "\n";
//            if ( res == data.yTrain[sampleId] ) {
//                train_good_counter += 1;
//            }
////            std::cout << res << " " << data.yTrain[sampleId] << "\n";
//
//            RelaxOutputLayer( output, target, OUTPUT_SIZE );
//            RelaxLayer( hidden2, HIDDEN2_SIZE );
//            RelaxLayer( hidden1, HIDDEN1_SIZE );
//            RelaxInputLayer( inputLayer );
//
//            GradStep( output, LEARNING_RATE_W, LEARNING_RATE_V );
//            GradStep( hidden2, LEARNING_RATE_W, LEARNING_RATE_V );
//            GradStep( hidden1, LEARNING_RATE_W, LEARNING_RATE_V );
//            GradStep( inputLayer, LEARNING_RATE_W, LEARNING_RATE_V );
//
//            ResetLayer( output );
//            ResetLayer( hidden2 );
//            ResetLayer( hidden1 );
//            ResetLayer( inputLayer );
//        }
//
//
//        int tp = 0;
//        int fp = 0;
//        int tn = 0;
//        int fn = 0;
//        float test_good_counter = 0;
//        for ( int sampleId = 0; sampleId < data.xTest.size(); sampleId++ ) {
//            EventManager eventManager( 50 );
//            RegisterSample( data.xTest[sampleId], eventManager, inputLayer );
//            eventManager.RunSimulation();
//
//            int t = data.yTest[sampleId];
//            Target target( OUTPUT_SIZE, 0 );
//            target[data.yTest[sampleId]] = 1;
//            float S = 0;
//            std::vector<float> softMax;
//            std::vector<float> deltas(OUTPUT_SIZE, 0);
//            float loss = SoftMaxLoss( output, target, softMax, deltas, &S );
//            totalTestLoss += loss;
////
//            int res = Argmax(softMax);
////            std::cout << softMax[0] << " " << softMax[1] << " " << softMax[2] << "\n";
//            if ( res == t ) {
//                test_good_counter += 1;
//            }
////            std::cout << res << " " << t << "\n";
//            ResetLayer( output );
//            ResetLayer( hidden2 );
//            ResetLayer( hidden1 );
//            ResetLayer( inputLayer );
//
//        }
//        std::cout << train_good_counter / data.yTrain.size() << " " << test_good_counter / data.yTest.size() << " ";
//        std::cout << " Average activity: " << train_activity / data.yTrain.size() << " ";
//        std::cout << " Average train loss: " << totalTrainLoss / data.yTrain.size();
//        std::cout << " Average test loss: " << totalTestLoss / data.yTest.size() << "\n";
//
//    }

}

void createSynapses(Layer &layer1, Layer &layer2,
                    int size1, int size2) {
    std::default_random_engine generator;
    generator.seed( clock() );
    float limit = sqrt( 2. / (  size1 * size2 ) );
    std::uniform_real_distribution<float> distribution( 0, limit );

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
//        *S += softMax[neuronId] * deltas[neuronId];
        loss += target[neuronId] * log(softMax[neuronId]);
    }
    return -loss;
}

void RelaxOutputLayer( Layer &outputLayer, const Target &target, int size ) {
    float Sa = 0;
    float S = 0;
    std::vector<float> softMax;
    std::vector<float> deltas(size, 0);
    for ( int neuronId = 0; neuronId < size; neuronId++ ) {
        outputLayer[neuronId].get()->RelaxOutput(49);
    }
    SoftMaxLoss( outputLayer, target, softMax, deltas, &S );
    for ( int softMaxId = 0; softMaxId < size; softMaxId++ ) {
        outputLayer[softMaxId].get()->grad = deltas[softMaxId];
        outputLayer[softMaxId].get()->RelaxOutput(49);
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
        neuron.get()->RelaxOutput(49);
        Sa += neuron.get()->a;
    }
    for ( auto neuron: layer ) {
        neuron.get()->Backward(Sa);
    }
}

void RelaxInputLayer( Layer &layer) {
    float Sa = 0;
    for ( auto neuron: layer ) {
        neuron.get()->RelaxOutput(49);
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

void GradStep( Layer &layer, float learningRateW, float learningRateV) {
    float N = layer.size(), M = 0, m = 0;
//    int N2 = layer[0].get()->outputSynapses.size();
//    if ( N2 == 0 ) {
//        N2 = 1;
//    }
    float S = 0;
    for ( auto neuron: layer ) {
        for ( auto synapse: neuron.get()->outputSynapses ) {
            if ( synapse.updatable ) {
                if ( neuron.get()->a > 0 ) {
                    m += 1;
                    S += synapse.strength * synapse.strength;
                }
                M += 1;
            }
        }
    }
    S = BETA * (S - 1);
    float F = exp(S);
    if ( m == 0 ) {
        m = 1;
    }
    if ( M == 0 ) {
        M = 1;
    }
//    int M = 1, N = 1, m = 1;
    for ( auto neuron: layer ) {
        neuron.get()->vMaxThresh -= neuron.get()->DlDV * learningRateV; //* sqrt(N / M / m);
        if ( neuron.get()->vMaxThresh < 0 ) {
            neuron.get()->vMaxThresh = 0.001;
        }
        std::vector<Synapse> &synapses = neuron.get()->outputSynapses;
        for ( int synapseId = 0; synapseId < synapses.size(); synapseId++ ) {
            if ( synapses[synapseId].updatable ) {
                synapses[synapseId].strength -= synapses[synapseId].DlDw * learningRateW;// * sqrt(N / m);
                if ( synapses[synapseId].strength < 0 ) {
                    synapses[synapseId].strength = 0;
                }
//                synapses[synapseId].strength -= BETA * LAMBDA * synapses[synapseId].strength * F;
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
                input[neuronId].get()->RelaxOutput( tick, true );
            }
        }
    }
    return sampleSize;
}

void InitLayer( Layer &layer, int size, int nextLayerSize ) {
    float limit;
    if ( nextLayerSize ) {
        limit = 0.2 * sqrt(3. / nextLayerSize );
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

template <class T> void PrintVector( const std::vector<T> &vector ) {
    std::cout << "Vector of size: " << vector.size() << "\n";
    for ( auto element: vector ) {
        std::cout << element << " ";
    }
    std::cout << "\n";
}



void TestForward() {

    float EPS = 1e-4;

    Layer inputLayer;
    InitLayer( inputLayer, 1, 1 );
    Layer hidden1;
    InitLayer( hidden1, 1, 1);
    Layer output;
    InitLayer( output, 1, 0);

    createSynapses(inputLayer, hidden1, 1, 1);
    createSynapses(hidden1, output, 1, 1 );
    output[0].get()->sigma_mu = 0;
    inputLayer[0].get()->sigma_mu = 0;
    hidden1[0].get()->sigma_mu = 0;

    hidden1[0].get()->vMaxThresh = 1.2;
    output[0].get()->vMaxThresh = 1.6;
    inputLayer[0].get()->outputSynapses[0].strength = 0.8;
    hidden1[0].get()->outputSynapses[0].strength = 0.9;


    Dataset data;
    Activity targetActivity;
    GenerateTestData( data, targetActivity );

    for ( int epochId = 0; epochId < 1; epochId++ ) {
        for ( int sampleId = 0; sampleId < data.xTrain.size(); sampleId++ ) {
            EventManager eventManager( 50 );
            int sampleSize = RegisterSample( data.xTrain[sampleId], eventManager, inputLayer );
            eventManager.RunSimulation();
            output.back().get()->RelaxOutput(49);
            float activityDelta = abs(output.back().get()->a - targetActivity.expectedOutput[sampleId]);
            if ( activityDelta < EPS ) {
                std::cout << "Forward test " << sampleId << " Successfully passed, status: OK\n";
            } else {
                std::cout << "Forward test " << sampleId << " Failed, status: Failed\n";
            }

            ResetLayer( output );
            ResetLayer( hidden1 );
            ResetLayer( inputLayer );
        }
    }
}

void TestBackProp() {
    Layer inputLayer;
    InitLayer( inputLayer, 2, 2 );
    Layer hidden1;
    InitLayer( hidden1, 2, 2 );
    Layer output;
    InitLayer( output, 2, 0);

    createSynapses(inputLayer, hidden1, 2, 2);
    createSynapses(hidden1, output, 2, 2);
    for ( int neuronId = 0; neuronId < OUTPUT_SIZE; neuronId++ ) {
        output[neuronId].get()->sigma_mu = 0;
    }
    for ( int neuronId = 0; neuronId < INPUT_SIZE; neuronId++ ) {
        inputLayer[neuronId].get()->sigma_mu = 0;
    }
    for ( int neuronId = 0; neuronId < HIDDEN1_SIZE; neuronId++ ) {
        hidden1[neuronId].get()->sigma_mu = 0;
    }

    Dataset data;
    Activity targetActivity;
    GenerateBackPropTestData(data, targetActivity);

    float EPS = 1e-1;
    float LEARNING_RATE_W = 0.01;
    float LEARNING_RATE_V = 0.0;

//    inputLayer[0].get()->outputSynapses[0].strength = 1;
//    inputLayer[0].get()->outputSynapses[1].strength = 0;
//    inputLayer[1].get()->outputSynapses[0].strength = 0;
//    inputLayer[1].get()->outputSynapses[1].strength = 1;
//
//    hidden1[0].get()->outputSynapses[0].strength = 0.01;
//    hidden1[0].get()->outputSynapses[1].strength = 0.02;
//    hidden1[1].get()->outputSynapses[0].strength = 0.03;
//    hidden1[1].get()->outputSynapses[1].strength = 0.04;

    for ( int epochId = 0; epochId < 50; epochId++ ) {
        for ( int sampleId = 0; sampleId < data.xTrain.size(); sampleId++ ) {
            EventManager eventManager( 50 );
            int sampleSize = RegisterSample( data.xTrain[sampleId], eventManager, inputLayer );
            eventManager.RunSimulation();
            Target target( 2, 0 );
            target[0] = data.yTrain[sampleId];
            target[1] = 1 - data.yTrain[sampleId];

            RelaxOutputLayer( output, target, 2 );
            RelaxLayer( hidden1, 2 );
            RelaxInputLayer( inputLayer );

            GradStep( output, LEARNING_RATE_W, LEARNING_RATE_V );
            GradStep( hidden1, LEARNING_RATE_W, LEARNING_RATE_V );
            GradStep( inputLayer, LEARNING_RATE_W, LEARNING_RATE_V );

            float S = 0;
            std::vector<float> softMax;
            std::vector<float> deltas(OUTPUT_SIZE, 0);
            float loss = SoftMaxLoss( output, target, softMax, deltas, &S );

            float activityDelta = deltas[0];
            if ( activityDelta < EPS ) {
                std::cout << "Forward test " << sampleId << " Successfully passed, status: OK\n";
            } else {
                std::cout << "Forward test " << sampleId << " Failed, status: Failed\n";
            }

            ResetLayer( output );
            ResetLayer( hidden1 );
            ResetLayer( inputLayer );
        }
    }
//    float averageDelta = 0;
//    for ( int sampleId = 0; sampleId < data.xTrain.size(); sampleId++ ) {
//        output.back().get()->RelaxOutput( 49 );
//        float activityDelta = abs( output.back().get()->a - targetActivity.expectedOutput[sampleId] );
//        averageDelta += activityDelta;
//        if ( activityDelta < EPS ) {
//            std::cout << "Forward test " << sampleId << " Successfully passed, status: OK\n";
//        } else {
//            std::cout << "Forward test " << sampleId << " Failed, status: Failed\n";
//        }
//    }
//    averageDelta /= data.xTrain.size();
}

void GenerateTestData( Dataset &data, Activity &targetActivity ) {
    int simulationTime = 50;
    int inputSize = 1;
    data.xTrain = std::vector<SpikeTrain>(4, SpikeTrain(simulationTime, std::vector<int>(inputSize)));
    for ( int timeId = 0; timeId < simulationTime; timeId++ ) {
        for ( int neuronId = 0; neuronId < inputSize; neuronId++ ) {
            data.xTrain[0][timeId][neuronId] = 1;
        }
    }
    for ( int timeId = 0; timeId < simulationTime; timeId++ ) {
        for ( int neuronId = 0; neuronId < inputSize; neuronId++ ) {
            data.xTrain[1][timeId][neuronId] = timeId % 2;
        }
    }
    for ( int timeId = 0; timeId < simulationTime; timeId++ ) {
        for ( int neuronId = 0; neuronId < inputSize; neuronId++ ) {
            data.xTrain[2][timeId][neuronId] = (timeId % 3 == 0);
        }
    }
    for ( int timeId = 0; timeId < simulationTime; timeId++ ) {
        for ( int neuronId = 0; neuronId < inputSize; neuronId++ ) {
            data.xTrain[3][timeId][neuronId] = 0;
        }
    }
    targetActivity.expectedOutput = std::vector<float>(4);
    targetActivity.expectedOutput[0] = 3.17442;
    targetActivity.expectedOutput[1] = 2.36357;
    targetActivity.expectedOutput[2] = 1.67968;
    targetActivity.expectedOutput[3] = 0;
}

void GenerateBackPropTestData( Dataset &data, Activity &targetActivity ) {
    int simulationTime = 50;
    int inputSize = 2;
    data.xTrain = std::vector<SpikeTrain>(3, SpikeTrain(simulationTime, std::vector<int>(inputSize)));
    for ( int timeId = 0; timeId < simulationTime; timeId++ ) {
        data.xTrain[0][timeId][0] = 1;
    }
    for ( int timeId = 0; timeId < simulationTime; timeId++ ) {
        data.xTrain[1][timeId][1] = 1;
    }
    for ( int timeId = 0; timeId < simulationTime; timeId++ ) {
        for ( int neuronId = 0; neuronId < inputSize; neuronId++ ) {
            data.xTrain[2][timeId][neuronId] = 1;
        }
    }
    data.yTrain = std::vector<float>(3);
    data.yTrain[0] = 1;
    data.yTrain[1] = 0;
    data.yTrain[2] = 0.5;
    targetActivity.expectedOutput = std::vector<float>(3);
    targetActivity.expectedOutput[0] = 1;
    targetActivity.expectedOutput[1] = 0;
    targetActivity.expectedOutput[2] = 0.5;
}

void GeneratePoissonSeries( std::vector<int> &data, int averageNumSpikes, int simulationTime ) {
    std::default_random_engine generator;
    generator.seed( clock() );
    std::uniform_real_distribution<float> distribution( 0, 1 );
    if ( simulationTime == 0 ) {
        simulationTime = data.size();
    }
    float threshold = static_cast<float>(averageNumSpikes) / simulationTime;
    for ( int index = 0; index <= simulationTime; ++index ) {
        data[index] = static_cast<int>(distribution(generator) < threshold);
    }

}