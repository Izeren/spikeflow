#include "EventManager.h"
#include "LifNeuron.h"
#include "Synapse.h"
#include <iostream>

void EventManager::RegisterSpikeEvent( Event event, Time time ) {
    eventsTimeline[time].push_back(event);
}

void EventManager::RunSimulation() {
    for ( Time time = 0; time < simulationTime; time++ ) {
        for ( Event event: eventsTimeline[time] ) {
            eventCounter += 1;
            auto synapses = event->GetOutputSynapses();
            for ( const Synapse *synapse: synapses ) {
                LifNeuron& next = synapse->GetNext();
                if ( next.IsConsistent() ) {
                    validationTimeline[time].push_back(&next);
                }
                next.UpdatePotential(time, synapse->GetStrength());
            }
        }
        for ( ValidationCandidate candidate: validationTimeline[time] ) {
            bool isActive = candidate->NormalizePotential(time);
            if ( isActive ) {
                if ( time + 1 < simulationTime ) {}
                RegisterSpikeEvent(candidate, time + 1);
            }
        }
    }
}

EventManager::EventManager( int simulationTime ) : simulationTime( simulationTime ) {
    eventsTimeline = EventsTimeline(simulationTime);
    validationTimeline = ValidationTimeline(simulationTime);
    eventCounter = 0;
}
