//  DAEDALUS – Distributed and Automated Evolutionary Deep Architecture Learning with Unprecedented Scalability
// 
// This research code was developed as part of the research programme Open Technology Programme with project number 18373, which was financed by the Dutch Research Council (NWO), Elekta, and Ortec Logiqcare.
// 
// Project leaders: Peter A.N. Bosman, Tanja Alderliesten
// Researchers: Alex Chebykin, Arthur Guijt, Vangelis Kostoulas
// Main code developer: Arthur Guijt

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
