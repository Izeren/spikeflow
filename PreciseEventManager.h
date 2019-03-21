#pragma once


#pragma once
#include <vector>

class LifNeuron;

typedef LifNeuron *Event;
typedef std::vector<Event> EventsList;
typedef LifNeuron *ValidationCandidate;
typedef std::vector<ValidationCandidate> ValidationCandidates;
typedef std::vector<EventsList> EventsTimeline;
typedef std::vector<ValidationCandidates> ValidationTimeline;
typedef float Time;

class PreciseEventManager {
public:
    void RegisterSpikeEvent(Event event, Time time);
    void RunSimulation();
    PreciseEventManager( int simulationTime );

    int eventCounter;

private:
    int numberOfTimeBuckets;
    EventsTimeline eventsTimeline;
    ValidationTimeline validationTimeline;

};