cmake_minimum_required(VERSION 3.13)
project(SpikeProp)

set(CMAKE_CXX_STANDARD 14)


set(
        DATASET_UTILS_FILES
        dataset_utils/IrisUtils.h
        dataset_utils/IrisUtils.cpp
        dataset_utils/MnistUtils.hpp
        dataset_utils/DataCommon.cpp dataset_utils/DataCommon.h)
include_directories( dataset_utils )
add_library( DatasetUtils SHARED ${DATASET_UTILS_FILES})


set(
        COMMON_UTILS_FILES
        common/SpikingGeneral.h
        common/SpikingGeneral.cpp
        common/INeuron.h
        common/INeuron.cpp
        synapses/ISynapse.h
        synapses/ISynapse.cpp
        common/INetwork.h
        common/INetwork.cpp
        common/BasicNetwork.hpp
        common/IEventManager.h
        stats/LayerStats.cpp stats/LayerStats.h stats/Stat.h layers/ILayer.cpp layers/ILayer.h losses/ILossFunction.cpp losses/ILossFunction.h layers/DenseLifLayer.cpp layers/DenseLifLayer.h neurons/INeuronBuilder.h layers/DenseVanillaLayer.cpp layers/DenseVanillaLayer.h common/scripts/DemoScripts.cpp common/scripts/DemoScripts.h)
include_directories( common )
add_library( Common SHARED ${COMMON_UTILS_FILES})


set (
        EVENT_MANAGER_FILES
        event_managers/PreciseEventManager.h
        event_managers/PreciseEventManager.cpp
)
include_directories( event_managers )
add_library( EventManagers SHARED ${EVENT_MANAGER_FILES} )


set (
        SYNAPSE_FILES
        synapses/LifSynapse.h
        synapses/LifSynapse.cpp
        synapses/BasicSynapse.cpp synapses/BasicSynapse.h synapses/ISynapseBuilder.h synapses/LifSynapseBuilder.cpp synapses/LifSynapseBuilder.h synapses/ISynapseBuilder.cpp synapses/VanillaSynapseBuilder.cpp synapses/VanillaSynapseBuilder.h)
include_directories( synapses )
add_library( Synapses SHARED ${SYNAPSE_FILES} )


set (
        NEURON_FILES
        neurons/LifNeuron.h
        neurons/LifNeuron.cpp
        neurons/BasicNeuron.cpp neurons/BasicNeuron.h neurons/LifNeuronBuilder.cpp neurons/LifNeuronBuilder.h neurons/BasicNeuronBuilder.cpp neurons/BasicNeuronBuilder.h)
include_directories( neurons )
add_library( Neurons SHARED ${NEURON_FILES} )


add_executable( SpikeProp main.cpp )


target_link_libraries( DatasetUtils Common )
target_link_libraries( Synapses Common )
target_link_libraries( Neurons Common )
target_link_libraries( EventManagers Common )

target_link_libraries( SpikeProp Common )
target_link_libraries( SpikeProp DatasetUtils )
target_link_libraries( SpikeProp EventManagers )
target_link_libraries( SpikeProp Synapses )
target_link_libraries( SpikeProp Neurons )

