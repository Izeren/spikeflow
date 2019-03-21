#include "PreciseEventManager.h"
#include "LifNeuron.h"


void PreciseEventManager::RegisterSpikeEvent( Event event, Time time ) {
    eventsTimeline[time].push_back(event);
}

void PreciseEventManager::RunSimulation() {
    for ( Time time = 0; time < simulationTime; time++ ) {
        for ( Event event: eventsTimeline[time] ) {
            eventCounter += 1;
            auto synapses = event->outputSynapses;
            if ( event->v > event->vMaxThresh ) {
                if ( time + 1 < simulationTime ) {
                    RegisterSpikeEvent( event, time + 1);
                }
            }
            for ( int synapseId = 0; synapseId < synapses.size(); synapseId++ ) {
                LifNeuron *next = synapses[synapseId].next;
                if ( next->IsConsistent()) {
                    validationTimeline[time].push_back(next);
                }
                next->UpdatePotential(time, synapses[synapseId].strength);
            }
        }
        for ( ValidationCandidate candidate: validationTimeline[time] ) {
            bool isActive = candidate->NormalizePotential(time);
            if ( isActive ) {
                if ( time + 1 < simulationTime ) {
                    RegisterSpikeEvent( candidate, time + 1 );
                }
            }
        }
    }
}

PreciseEventManager::PreciseEventManager( int simulationTime ) : simulationTime( simulationTime ) {
    eventsTimeline = EventsTimeline(simulationTime);
    validationTimeline = ValidationTimeline(simulationTime);
    eventCounter = 0;
}
