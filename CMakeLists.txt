cmake_minimum_required(VERSION 3.13)
project(SpikeProp)

set(CMAKE_CXX_STANDARD 14)


set(
        DATASET_UTILS_FILES
        dataset_utils/IrisUtils.h
        dataset_utils/IrisUtils.cpp
        dataset_utils/MnistUtils.hpp
)
include_directories( dataset_utils )
add_library( DatasetUtils SHARED ${DATASET_UTILS_FILES})


set(
        COMMON_UTILS_FILES
        common/SpikingGeneral.h
        common/SpikingGeneral.cpp
        common/INeuron.h
        common/INeuron.cpp
        common/ISynapse.h
        common/ISynapse.cpp
        common/INetwork.h
        common/INetwork.cpp
        common/BasicNetwork.hpp
        common/IEventManager.h
        common/IEventManager.cpp
)
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
        synapses/Synapse.h
        synapses/Synapse.cpp
)
include_directories( synapses )
add_library( Synapses SHARED ${SYNAPSE_FILES} )


set (
        NEURON_FILES
        neurons/LifNeuron.h
        neurons/LifNeuron.cpp
)
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

