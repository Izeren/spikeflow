# Spikeflow
This framework is my attempt to create common framework for Spiking Neural Networks with as wide as possible 
spectrum of possible experiments.


## Very ALPHA stage. 

It is not easy task to fix architecture now, making it fast enough and the same time as flexible as it can be. 
Current implementation is targeted on experiments which can have multiple synaptic links with different time
delays and also recurrent architectures, that's why I prefer to use event driven logic. 

## Event driven logic

Main implementation here uses event driven calculations. This enables precise time discretization and reduces 
the number of required calculations for the sparce and partially sparse models.

## Branches of development

Learning of spiking neural networks mainly separated into 4 main approaches:

1. Hebbian based rules: STDP, R-STDP and so on.
2. Spike propagation: the attempt to adopt gradient descent to the spiking nature.
3. Neuroevolution: NEAT, hyperNEAT and so on.
4. Weights transfer: mainly based on frequency encoding and pretrained ReLU networks.

In this framework, I would like to introduce joint possibility to use 1 and 3 together. Maybe even add the 
Spikeprop support in the future. 
