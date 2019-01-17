#pragma once
#include <vector>

class LifNeuron;

typedef LifNeuron *Event;
typedef std::vector<Event> EventsList;
typedef LifNeuron *ValidationCandidate;
typedef std::vector<ValidationCandidate> ValidationCandidates;
typedef std::vector<EventsList> EventsTimeline;
typedef std::vector<ValidationCandidates> ValidationTimeline;
typedef int Time;

class EventManager {
public:
    void RegisterSpikeEvent(Event event, Time time);
    void RunSimulation();
    EventManager( int simulationTime );

    int eventCounter;

private:
    int simulationTime;
    EventsTimeline eventsTimeline;
    ValidationTimeline validationTimeline;

};