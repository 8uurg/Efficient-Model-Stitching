//  DAEDALUS – Distributed and Automated Evolutionary Deep Architecture Learning with Unprecedented Scalability
// 
// This research code was developed as part of the research programme Open Technology Programme with project number 18373, which was financed by the Dutch Research Council (NWO), Elekta, and Ortec Logiqcare.
// 
// Project leaders: Peter A.N. Bosman, Tanja Alderliesten
// Researchers: Alex Chebykin, Arthur Guijt, Vangelis Kostoulas
// Main code developer: Arthur Guijt

#include "running.hpp"
#include "base.hpp"

TerminationStepper::TerminationStepper(std::function<std::shared_ptr<GenerationalApproach>()> approach,
                                       std::optional<int> step_limit) :
    approach(approach()), step_limit(step_limit)
{
}
void TerminationStepper::setPopulation(std::shared_ptr<Population> population)
{
    IRunner::setPopulation(population);
    approach->setPopulation(population);
}
void TerminationStepper::registerData()
{
    approach->registerData();
}
void TerminationStepper::afterRegisterData()
{
    approach->afterRegisterData();
}

void TerminationStepper::step()
{
    if (!terminated && (!step_limit.has_value() || *step_limit > steps))
    try
    {
        approach->step();
        ++steps;
    }
    catch (run_complete &e)
    {
        terminated = true;
        throw;
    }
    catch (stop_approach &e)
    {
        terminated = true;
        throw e;
    }
}

void TerminationStepper::run()
{
    try
    {
        while (!step_limit.has_value() || *step_limit > steps)
        {
            approach->step();
            ++steps;
            if (approach->terminated()) break;
        }
    }
    catch (time_limit_reached &e)
    {
    }
    catch (evaluation_limit_reached &e)
    {
    }
    catch (vtr_reached &e)
    {
    }
    catch (stop_approach &e)
    {
    }
}

void InterleavedMultistartScheme::run()
{
    try
    {
        while (true)
        {
            auto new_population = approach_factory(current);
            new_population->setPopulation(this->population);
            if (approaches.size() == 0)
            {
                new_population->registerData();
                new_population->afterRegisterData();
            }

            approaches.push_back(std::move(new_population));
            current *= multiplier;
            runRecursiveFold(approaches.size() - 1);
        }
    }
    catch (run_complete &e)
    {
    }
    catch (stop_approach &e)
    {
    }
}
void InterleavedMultistartScheme::step()
{
    if (!terminated)
    try
    {
        auto new_population = approach_factory(current);
        new_population->setPopulation(this->population);
        if (approaches.size() == 0)
        {
            new_population->registerData();
            new_population->afterRegisterData();
        }

        approaches.push_back(std::move(new_population));
        current *= multiplier;
        runRecursiveFold(approaches.size() - 1);
    }
    catch (time_limit_reached &e)
    {
        terminated = true;
        throw e;
    }
    catch (evaluation_limit_reached &e)
    {
        terminated = true;
        throw e;
    }
    catch (vtr_reached &e)
    {
        terminated = true;
        throw e;
    }
    catch (stop_approach &e)
    {
        terminated = true;
        throw e;
    }
}
void InterleavedMultistartScheme::runRecursiveFold(size_t end_index)
{
    // Base case.
    if (minimum_index > end_index)
        return;

    for (size_t i = 0; i < steps - 1; ++i)
    {
        updateMinimumIndex();
        for (size_t i = minimum_index; i <= end_index; ++i)
        {
            approaches[i]->step();
        }

        for (size_t i = minimum_index; i < end_index; ++i)
        {
            runRecursiveFold(i);
        }
    }
}

InterleavedMultistartScheme::InterleavedMultistartScheme(
    std::function<std::shared_ptr<GenerationalApproach>(size_t)> approach_factory,
    std::shared_ptr<GenerationalApproachComparator> approach_comparator,
    size_t steps,
    size_t base,
    size_t multiplier) :
    steps(steps),
    current(base),
    multiplier(multiplier),
    approach_factory(approach_factory),
    approach_comparator(approach_comparator)
{
}

void InterleavedMultistartScheme::updateMinimumIndex()
{
    // Clear any caches, might have had some remaining from previous updates.
    approach_comparator->clear();

    while (true)
    {
        bool any_better = false;

        for (size_t other = minimum_index + 1; other < approaches.size(); ++other)
        {
            short c = approach_comparator->compare(approaches[minimum_index], approaches[other]);
            if (c == 2) // other is better
            {
                any_better = true;
                break;
            }
        }

        if (! any_better) break;
        // Increase current minimum index, other population is expected to improve further
        // and is already outperforming current. So providing any budget to this index is no longer neccesary.
        ++minimum_index;
    }
}

void InterleavedMultistartScheme::setPopulation(std::shared_ptr<Population> population)
{
    this->population = population;
    approach_comparator->setPopulation(population);
}
void InterleavedMultistartScheme::registerData()
{
    approach_comparator->registerData();
}
void InterleavedMultistartScheme::afterRegisterData()
{
    approach_comparator->afterRegisterData();
}

SimpleConfigurator::SimpleConfigurator(std::shared_ptr<ObjectiveFunction> objective,
                                       std::shared_ptr<IRunner> runner,
                                       std::optional<size_t> seed) :
    objective(objective), runner(std::move(runner)), rng(seed)
{
    population = std::make_shared<Population>();
    population->registerGlobalData(GObjectiveFunction(&*objective));
    population->registerGlobalData(rng);
    this->runner->setPopulation(population);
    this->objective->setPopulation(population);
    this->runner->registerData();
    this->objective->registerData();
    this->runner->afterRegisterData();
    this->objective->afterRegisterData();
}

void SimpleConfigurator::run()
{
    runner->run();
}


void SimpleConfigurator::step()
{
    runner->step();
}

