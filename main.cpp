#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include "LifNeuron.h"
#include "Synapse.h"
#include "EventManager.h"

#include <eigen3/Eigen/Dense>

using Eigen::MatrixXd;

const int INPUT_SIZE = 4;
const int HIDDEN1_SIZE = 4;
const int HIDDEN2_SIZE = 3;
const int OUTPUT_SIZE = 3;



void parseIris();

int main() {
//    parseIris();
    EventManager eventManager(50);
    LifNeuron inputNeuron(std::vector<Synapse *>(), 0, 0, -5, 10, 20, -1, -1, 2, true);
    LifNeuron hiddenNeuron(std::vector<Synapse *>(), 0, 0, -5, 10, 20, -1, -1, 100, true);
    LifNeuron outputNeuron(std::vector<Synapse *>(), 0, 0, -5, 10, 20, -1, -1, 2, true);
    Synapse synapse1(true, 5, 0, inputNeuron, hiddenNeuron);
    Synapse synapse2(true, 5, 0, hiddenNeuron, outputNeuron);
    inputNeuron.AddSynapse(&synapse1);
    hiddenNeuron.AddSynapse(&synapse2);

    eventManager.RegisterSpikeEvent(&inputNeuron, 0);
    eventManager.RegisterSpikeEvent(&inputNeuron, 10);
    eventManager.RegisterSpikeEvent(&inputNeuron, 20);
    eventManager.RegisterSpikeEvent(&inputNeuron, 30);
    eventManager.RegisterSpikeEvent(&inputNeuron, 40);

    eventManager.RunSimulation();
    std::cout << eventManager.eventCounter << "\n";

}


//void parseIris() {
//    std::ifstream ifs("/home/izeren/CLionProjects/SpikeProp/iris.data");
//    std::string tmp;
//    int sample_id = 0;
//    while (std::getline(ifs, tmp)) {
//        std::stringstream ss(tmp);
//        float param;
//        char sep;
//        for (int param_id = 0; param_id < 4; param_id++) {
//            ss >> data.x(sample_id, param_id);
//            ss >> sep;
//        }
//        ss >> tmp;
//        if (tmp == "Iris-setosa") {
//            data.y(sample_id, 0) = 0;
//        } else if (tmp == "Iris-versicolor") {
//            data.y(sample_id, 0) = 1;
//        } else {
//            data.y(sample_id, 0) = 2;
//        }
//        sample_id++;
//    }
//}