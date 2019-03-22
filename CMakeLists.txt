cmake_minimum_required(VERSION 3.13)
project(SpikeProp)

set(CMAKE_CXX_STANDARD 14)

add_executable(SpikeProp main.cpp Synapse.h LifNeuron.cpp Synapse.cpp EventManager.h EventManager.cpp MnistUtils.hpp PreciseEventManager.h PreciseEventManager.cpp SpikingGeneral.h SpikingGeneral.cpp INeuron.cpp INeuron.h ISynapse.cpp ISynapse.h INetwork.cpp INetwork.h)