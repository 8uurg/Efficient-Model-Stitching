#include "base.hpp"
#include "sim.hpp"

Objective& getObjective(Population &pop, Individual &i)
{
    return pop.getData<Objective>(i);
}

GenotypeCategorical& getGenotypeCategorical(Population &pop, Individual &i)
{
    return pop.getData<GenotypeCategorical>(i);
}

TimeSpent& getTimeSpent(Population &pop, Individual &i)
{
    return pop.getData<TimeSpent>(i);
}
