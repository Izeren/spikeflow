#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include "LifNeuron.h"
//#include "Synapse.h"

#include <eigen3/Eigen/Dense>

using Eigen::MatrixXd;

const int INPUT_SIZE = 4;
const int HIDDEN1_SIZE = 4;
const int HIDDEN2_SIZE = 3;
const int OUTPUT_SIZE = 3;

struct {
    MatrixXd v;
    MatrixXd vThresh;
    MatrixXd a;
    MatrixXd mu;
    MatrixXd tps;
    MatrixXd tout;
} input={ MatrixXd(INPUT_SIZE, 1), MatrixXd(INPUT_SIZE, 1), MatrixXd(INPUT_SIZE, 1), MatrixXd(INPUT_SIZE, 1),
          MatrixXd(INPUT_SIZE, 1), MatrixXd(INPUT_SIZE, 1)};

struct {
    MatrixXd v;
    MatrixXd vThresh;
    MatrixXd a;
    MatrixXd x;
    MatrixXd mu;
    MatrixXd tps;
    MatrixXd tout;
} hidden1={ MatrixXd(HIDDEN1_SIZE, 1), MatrixXd(HIDDEN1_SIZE, 1), MatrixXd(HIDDEN1_SIZE, 1),
            MatrixXd(HIDDEN1_SIZE, 1), MatrixXd(HIDDEN1_SIZE, 1),
            MatrixXd(HIDDEN1_SIZE, 1), MatrixXd(HIDDEN1_SIZE, 1) };

struct {
    MatrixXd v;
    MatrixXd vThresh;
    MatrixXd a;
    MatrixXd x;
    MatrixXd mu;
    MatrixXd tps;
    MatrixXd tout;
} hidden2={ MatrixXd(HIDDEN2_SIZE, 1), MatrixXd(HIDDEN2_SIZE, 1), MatrixXd(HIDDEN2_SIZE, 1),
            MatrixXd(HIDDEN2_SIZE, 1), MatrixXd(HIDDEN2_SIZE, 1),
            MatrixXd(HIDDEN2_SIZE, 1), MatrixXd(HIDDEN2_SIZE, 1) };

struct {
    MatrixXd v;
    MatrixXd vThresh;
    MatrixXd a;
    MatrixXd mu;
    MatrixXd tps;
    MatrixXd tout;
} output={ MatrixXd(OUTPUT_SIZE, 1), MatrixXd(OUTPUT_SIZE, 1), MatrixXd(OUTPUT_SIZE, 1), MatrixXd(OUTPUT_SIZE, 1),
           MatrixXd(OUTPUT_SIZE, 1), MatrixXd(OUTPUT_SIZE, 1)};


struct {
    MatrixXd w1;
    MatrixXd w2;
    MatrixXd w3;
} synapses = { MatrixXd(INPUT_SIZE, HIDDEN1_SIZE), MatrixXd(HIDDEN1_SIZE, HIDDEN2_SIZE),
               MatrixXd(HIDDEN2_SIZE, OUTPUT_SIZE) };

struct {
    MatrixXd g1;
    MatrixXd g2;
    MatrixXd g3;
} gradInputs = { MatrixXd(HIDDEN1_SIZE, 1), MatrixXd(HIDDEN2_SIZE, 1), MatrixXd(OUTPUT_SIZE, 1)} ;

struct {
    MatrixXd g1;
    MatrixXd g2;
    MatrixXd g3;
} gradW = { MatrixXd(INPUT_SIZE, HIDDEN1_SIZE), MatrixXd(HIDDEN1_SIZE, HIDDEN2_SIZE),
            MatrixXd(HIDDEN2_SIZE, OUTPUT_SIZE) };

float getWDyn(float tout, float tp, float tRef) {
    if (tout < 0 || tp < 0 || tRef < 0 || tp - tout < tRef) {
        return 1;
    } else {
        return (tp - tout) * (tp - tout) / tRef;
    }
}

struct {
    MatrixXd x;
    MatrixXd y;
} data = { MatrixXd(150, 4), MatrixXd(150, 1) };

void parseIris();

int main() {
    parseIris();

    std::cout << data.x << std::endl;
    std::cout << data.y << std::endl;
}


void parseIris() {
    std::ifstream ifs("/home/izeren/CLionProjects/SpikeProp/iris.data");
    std::string tmp;
    int sample_id = 0;
    while (std::getline(ifs, tmp)) {
        std::stringstream ss(tmp);
        float param;
        char sep;
        for (int param_id = 0; param_id < 4; param_id++) {
            ss >> data.x(sample_id, param_id);
            ss >> sep;
        }
        ss >> tmp;
        if (tmp == "Iris-setosa") {
            data.y(sample_id, 0) = 0;
        } else if (tmp == "Iris-versicolor") {
            data.y(sample_id, 0) = 1;
        } else {
            data.y(sample_id, 0) = 2;
        }
        sample_id++;
    }
}