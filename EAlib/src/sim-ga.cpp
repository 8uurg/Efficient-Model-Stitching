#include "sim-ga.hpp"

// SimulatedSynchronousSimpleGA

SimulatedSynchronousSimpleGA::SimulatedSynchronousSimpleGA(std::shared_ptr<SimulatorParameters> sim,
                                                           size_t population_size,
                                                           size_t offspring_size,
                                                           int replacement_strategy,
                                                           std::shared_ptr<ISolutionInitializer> initializer,
                                                           std::shared_ptr<ICrossover> crossover,
                                                           std::shared_ptr<IMutation> mutation,
                                                           std::shared_ptr<ISelection> parent_selection,
                                                           std::shared_ptr<IPerformanceCriterion> performance_criterion,
                                                           std::shared_ptr<IArchive> archive,
                                                           std::shared_ptr<SpanLogger> sl) :
    sim(sim),
    initializer(initializer),
    crossover(crossover),
    mutation(mutation),
    parent_selection(parent_selection),
    performance_criterion(performance_criterion),
    archive(archive),
    sl(sl),
    population_size(population_size),
    offspring_size(offspring_size),
    replacement_strategy(replacement_strategy)
{
}
SimulatedSynchronousSimpleGA::SimulatedSynchronousSimpleGA(std::shared_ptr<SimulatorParameters> sim,
                                                           size_t population_size,
                                                           size_t offspring_size,
                                                           std::shared_ptr<ISolutionInitializer> initializer,
                                                           std::shared_ptr<ICrossover> crossover,
                                                           std::shared_ptr<IMutation> mutation,
                                                           std::shared_ptr<ISelection> parent_selection,
                                                           std::shared_ptr<IPerformanceCriterion> performance_criterion,
                                                           std::shared_ptr<IArchive> archive,
                                                           bool include_population,
                                                           std::shared_ptr<ISelection> generationalish_selection,
                                                           std::shared_ptr<SpanLogger> sl) :
    sim(sim),
    initializer(initializer),
    crossover(crossover),
    mutation(mutation),
    parent_selection(parent_selection),
    performance_criterion(performance_criterion),
    archive(archive),
    sl(sl),
    population_size(population_size),
    offspring_size(offspring_size),
    replacement_strategy(6),
    target_selection_pool_size(offspring_size),
    include_population(include_population),
    generationalish_selection(generationalish_selection)
{
}

void SimulatedSynchronousSimpleGA::initialize()
{
    individuals.resize(population_size);
    population->newIndividuals(individuals);
    offspring.resize(offspring_size);
    population->newIndividuals(offspring);
    initializer->initialize(individuals);
    for (size_t idx = 0; idx < individuals.size(); ++idx)
    {
        evaluate_initial_solution(idx, individuals[idx]);
    }
    sim->simulator->simulate_until_end();
}
void SimulatedSynchronousSimpleGA::replace_fifo(Individual ii)
{
    Population &pop = *population;
    if (sl != NULL)
    {
        sl->end_span(current_replacement_index, individuals[current_replacement_index], 0);
        sl->start_span(current_replacement_index, ii, 0);
    }
    pop.copyIndividual(ii, individuals[current_replacement_index]);
    current_replacement_index += 1;
    if (current_replacement_index >= population_size)
        current_replacement_index = 0;
}
void SimulatedSynchronousSimpleGA::replace_idx(size_t idx, Individual ii)
{
    Population &pop = *population;
    if (sl != NULL)
    {
        sl->end_span(idx, individuals[idx], 0);
        sl->start_span(idx, ii, 0);
    }
    pop.copyIndividual(ii, individuals[idx]);
}
void SimulatedSynchronousSimpleGA::replace_uniformly(Individual ii)
{
    Population &pop = *population;
    std::uniform_int_distribution<size_t> d_pop_idx(0, population_size - 1);
    size_t idx = d_pop_idx(cache->rng.rng);
    if (sl != NULL)
    {
        sl->end_span(idx, individuals[idx], 0);
        sl->start_span(idx, ii, 0);
    }
    pop.copyIndividual(ii, individuals[idx]);
}
void SimulatedSynchronousSimpleGA::replace_selection_fifo(Individual ii)
{
    Population &pop = *population;
    size_t idx = current_replacement_index++;
    short c = performance_criterion->compare(individuals[idx], ii);
    if ((c == 3 && replace_if_equal) || (c == 0 && replace_if_incomparable) || (c == 2))
    {
        if (sl != NULL)
        {
            sl->end_span(idx, individuals[idx], 0);
            sl->start_span(idx, ii, 0);
        }
        pop.copyIndividual(ii, individuals[idx]);
    }
    if (current_replacement_index >= population_size)
        current_replacement_index = 0;
}
void SimulatedSynchronousSimpleGA::replace_selection_idx(size_t idx, Individual ii)
{
    Population &pop = *population;
    short c = performance_criterion->compare(individuals[idx], ii);
    if ((c == 3 && replace_if_equal) || (c == 0 && replace_if_incomparable) || (c == 2))
    {
        if (sl != NULL)
        {
            sl->end_span(idx, individuals[idx], 0);
            sl->start_span(idx, ii, 0);
        }
        pop.copyIndividual(ii, individuals[idx]);
    }
}
void SimulatedSynchronousSimpleGA::replace_selection_uniformly(Individual ii)
{
    Population &pop = *population;
    std::uniform_int_distribution<size_t> d_pop_idx(0, population_size - 1);
    size_t idx = d_pop_idx(cache->rng.rng);
    short c = performance_criterion->compare(individuals[idx], ii);
    if ((c == 3 && replace_if_equal) || (c == 0 && replace_if_incomparable) || (c == 2))
    {
        if (sl != NULL)
        {
            sl->end_span(idx, individuals[idx], 0);
            sl->start_span(idx, ii, 0);
        }
        pop.copyIndividual(ii, individuals[idx]);
    }
}
void SimulatedSynchronousSimpleGA::contender_generational_like_selection(Individual ii)
{
    // Defer inclusion in population by adding to the selection pool
    // Note however: we need to copy the solution, as the current individual will
    // be resampled.
    Individual ii_c = population->newIndividual();
    population->copyIndividual(ii, ii_c);

    // Add copy as contender.
    selection_pool.push_back(ii_c);

    // Exit if it is not time to select yet.
    if (selection_pool.size() < target_selection_pool_size)
        return;

    // Add in the current population as individuals
    if (include_population)
    {
        for (size_t idx = 0; idx < individuals.size(); ++idx)
        {
            Individual cp_ii = population->newIndividual();
            population->copyIndividual(individuals[idx], cp_ii);
            selection_pool.push_back(cp_ii);
        }
    }

    // Perform selection!
    std::vector<Individual> selected = generationalish_selection->select(selection_pool, population_size);

    // Replace population with selected solutions
    for (size_t idx = 0; idx < individuals.size(); ++idx)
    {
        population->copyIndividual(selected[idx], individuals[idx]);
    }

    // Drop temporary solutions in selection pool
    for (auto &ii_s : selection_pool)
    {
        population->dropIndividual(ii_s);
    }
    selection_pool.resize(0);
}
void SimulatedSynchronousSimpleGA::place_in_population(size_t idx,
                                                       const Individual ii,
                                                       const std::optional<int> override_replacement_strategy)
{
    archive->try_add(ii);
    int replacement_strategy_to_use = replacement_strategy;
    if (override_replacement_strategy.has_value())
        replacement_strategy_to_use = *override_replacement_strategy;

    switch (replacement_strategy_to_use)
    {
    case 0:
        replace_fifo(ii);
        break;
    case 1:
        replace_idx(idx, ii);
        break;
    case 2:
        replace_uniformly(ii);
        break;
    case 3:
        replace_selection_fifo(ii);
        break;
    case 4:
        replace_selection_idx(idx, ii);
        break;
    case 5:
        replace_selection_uniformly(ii);
        break;
    case 6:
        // Population-based Gather-until-sufficient-then-select
        contender_generational_like_selection(ii);
        break;
    }
}
void SimulatedSynchronousSimpleGA::evaluate_initial_solution(size_t idx, Individual &ii)
{
    Population &pop = *population;
    GObjectiveFunction &objective_function = *pop.getGlobalData<GObjectiveFunction>();

    doCache();

    TimeSpent &ts = cache->tgts.getData(ii);
    ts.t = 0;
    std::optional<std::exception_ptr> maybe_exception;
    try
    {
        objective_function.of->evaluate(ii);
    }
    catch (std::exception &e)
    {
        maybe_exception = std::current_exception();
    }

    // Wait until a processor is available.
    sim->simulator->simulate_until([this]() { return sim->num_workers_busy < sim->num_workers; });

    ++sim->num_workers_busy;
    sim->simulator->insert_event(
        std::make_unique<FunctionalSimResumable>(
            [idx, ii, maybe_exception, this](ISimulator &, double, std::unique_ptr<IResumableSimulated> &) {
                // Perform replacement
                place_in_population(idx, ii, 1);
                --sim->num_workers_busy;
                if (maybe_exception.has_value())
                    std::rethrow_exception(*maybe_exception);
            }),
        sim->simulator->now() + ts.t,
        "New solution on " + std::to_string(idx));
}
void SimulatedSynchronousSimpleGA::evaluate_solution(size_t idx, Individual ii)
{
    Population &pop = *population;
    GObjectiveFunction &objective_function = *pop.getGlobalData<GObjectiveFunction>();

    TimeSpent &ts = cache->tgts.getData(ii);
    ts.t = 0;
    std::optional<std::exception_ptr> maybe_exception;
    try
    {
        objective_function.of->evaluate(ii);
    }
    catch (std::exception &e)
    {
        maybe_exception = std::current_exception();
    }

    // Wait until a processor is available.
    sim->simulator->simulate_until([this]() { return sim->num_workers_busy < sim->num_workers; });

    ++sim->num_workers_busy;
    sim->simulator->insert_event(
        std::make_unique<FunctionalSimResumable>(
            [idx, ii, maybe_exception, this](ISimulator &, double, std::unique_ptr<IResumableSimulated> &) {
                // Perform replacement
                place_in_population(idx, ii, std::nullopt);
                --sim->num_workers_busy;
                if (maybe_exception.has_value())
                    std::rethrow_exception(*maybe_exception);
            }),
        sim->simulator->now() + ts.t,
        "New solution on " + std::to_string(idx));
}
bool SimulatedSynchronousSimpleGA::terminated()
{
    // Not converged if we haven't started yet.
    if (!initialized)
        return false;

    Population &pop = *population;
    auto tggc = pop.getDataContainer<GenotypeCategorical>();
    auto &r = tggc.getData(individuals[0]);

    // This is sufficient for synchronous!
    for (Individual ii : individuals)
    {
        auto &o = tggc.getData(ii);
        if (!std::equal(r.genotype.begin(), r.genotype.end(), o.genotype.begin()))
        {
            return false;
        }
    }
    return true;
}
void SimulatedSynchronousSimpleGA::generation()
{
    if (terminated())
        throw stop_approach();

    for (size_t idx = 0; idx < offspring.size(); ++idx)
    {
        auto &nii = offspring[idx];
        sample_solution(idx, nii);
        evaluate_solution(idx, nii);
    }

    sim->simulator->simulate_until_end();
}
void SimulatedSynchronousSimpleGA::step()
{
    if (!initialized)
    {
        initialize();
        initialized = true;
    }
    else
    {
        generation();
    }
}
std::vector<Individual> SimulatedSynchronousSimpleGA::sample_solutions()
{
    size_t num_parents = crossover->num_parents();
    std::vector<Individual> parents = parent_selection->select(individuals, num_parents);
    auto c_offspring = crossover->crossover(parents);
    mutation->mutate(c_offspring);
    return c_offspring;
}
std::vector<Individual> &SimulatedSynchronousSimpleGA::getSolutionPopulation()
{
    return individuals;
}
void SimulatedSynchronousSimpleGA::setPopulation(std::shared_ptr<Population> population)
{
    GenerationalApproach::setPopulation(population);

    this->initializer->setPopulation(population);
    this->crossover->setPopulation(population);
    this->mutation->setPopulation(population);
    this->parent_selection->setPopulation(population);
    this->performance_criterion->setPopulation(population);
    this->archive->setPopulation(population);
    if (generationalish_selection != NULL)
        this->generationalish_selection->setPopulation(population);
    if (sl != NULL)
        this->sl->setPopulation(population);
}
void SimulatedSynchronousSimpleGA::registerData()
{
    GenerationalApproach::registerData();

    this->initializer->registerData();
    this->crossover->registerData();
    this->mutation->registerData();
    this->parent_selection->registerData();
    this->performance_criterion->registerData();
    this->archive->registerData();
    if (generationalish_selection != NULL)
        this->generationalish_selection->registerData();
    if (sl != NULL)
        this->sl->registerData();
}
void SimulatedSynchronousSimpleGA::afterRegisterData()
{
    GenerationalApproach::afterRegisterData();

    this->initializer->afterRegisterData();
    this->crossover->afterRegisterData();
    this->mutation->afterRegisterData();
    this->parent_selection->afterRegisterData();
    this->performance_criterion->afterRegisterData();
    this->archive->afterRegisterData();
    if (generationalish_selection != NULL)
        this->generationalish_selection->afterRegisterData();
    if (sl != NULL)
        this->sl->afterRegisterData();
}

// SimulatedAsynchronousSimpleGA

SimulatedAsynchronousSimpleGA::SimulatedAsynchronousSimpleGA(
    std::shared_ptr<SimulatorParameters> sim,
    size_t population_size,
    size_t offspring_size,
    int replacement_strategy,
    std::shared_ptr<ISolutionInitializer> initializer,
    std::shared_ptr<ICrossover> crossover,
    std::shared_ptr<IMutation> mutation,
    std::shared_ptr<ISelection> parent_selection,
    std::shared_ptr<IPerformanceCriterion> performance_criterion,
    std::shared_ptr<IArchive> archive,
    std::shared_ptr<SpanLogger> sl) :
    SimulatedSynchronousSimpleGA(sim,
                                 population_size,
                                 offspring_size,
                                 replacement_strategy,
                                 initializer,
                                 crossover,
                                 mutation,
                                 parent_selection,
                                 performance_criterion,
                                 archive,
                                 sl)
{
}
void SimulatedAsynchronousSimpleGA::initialize()
{
    Population &population = *this->population;

    individuals.resize(population_size);
    population.newIndividuals(individuals);
    initializer->initialize(individuals);

    offspring.resize(offspring_size);
    population.newIndividuals(offspring);

    for (size_t idx = 0; idx < individuals.size(); ++idx)
    {
        population.copyIndividual(individuals[idx], offspring[idx]);
        // Initial evaluations should always greedily replace the original individual
        // to simulate the actual completion of evaluation of this solution.
        evaluate_initial_solution(idx, offspring[idx]);
    }

    // sim->simulator->simulate_until_end();
}
void SimulatedAsynchronousSimpleGA::generation()
{
    if (terminated())
        throw stop_approach();

    for (size_t num = 0; num < population_size; ++num)
    {
        // t_assert(sim->num_workers_busy <= sim->num_workers,
        //          "Number of busy workers should be upper bounded by the number of workers.");
        t_assert(sim->processing_queue.size() > 0 || sim->simulator->event_queue.size() > 0,
                 "Either queue length or processing queue should have items inside.");
        while (sim->processing_queue.size() > 0 && sim->num_workers_busy < sim->num_workers)
        {
            auto &[t_cost, resumable, msg] = sim->processing_queue.front();
            // Note: we do not have a descriptive message to use here...
            // Maybe provide one in the processing queue?
            sim->simulator->insert_event(std::move(resumable), sim->simulator->now() + t_cost, msg);
            sim->processing_queue.pop();
            ++sim->num_workers_busy;
        }
        // bool bounded_before_step = sim->num_workers_busy <= sim->num_workers;
        sim->simulator->step();
        // bool bounded_after_step = sim->num_workers_busy <= sim->num_workers;
        // t_assert(bounded_after_step | !bounded_before_step,
        //          "Number of busy workers should be upper bounded by the number of workers, yet a step broke this.");
    }
}
void SimulatedAsynchronousSimpleGA::sample_and_evaluate_new_solution(size_t idx)
{
    Individual ii = offspring[idx];
    sample_solution(idx, ii);
    evaluate_solution(idx, ii);
}
void SimulatedAsynchronousSimpleGA::evaluate_solution(size_t idx, Individual ii)
{
    Population &pop = *population;
    GObjectiveFunction &objective_function = *pop.getGlobalData<GObjectiveFunction>();

    // Evaluate it & track time cost
    TimeSpent &ts = cache->tgts.getData(ii);
    ts.t = 0;
    std::optional<std::exception_ptr> maybe_exception;
    try
    {
        objective_function.of->evaluate(ii);
    }
    catch (std::exception &e)
    {
        maybe_exception = std::current_exception();
    }

    // Wait until a processor is available for replacement.
    // sim->simulator->simulate_until([this]() { return sim->num_workers_busy < sim->num_workers; });

    // Once a processor is available, start operating.
    // ++sim->num_workers_busy;
    sim->simulator->insert_event(
        std::make_unique<FunctionalSimResumable>(
            [maybe_exception, idx, ii, this](ISimulator &, double, std::unique_ptr<IResumableSimulated> &) {
                // Perform replacement in population.
                place_in_population(idx, ii, std::nullopt);
                // Free up the worker (simulated work has completed)
                --sim->num_workers_busy;
                sim->processing_queue.push(
                    std::make_tuple(0.0,
                                    std::make_unique<FunctionalSimResumable>(
                                        [this, idx](ISimulator &, double, std::unique_ptr<IResumableSimulated> &) {
                                            sample_and_evaluate_new_solution(idx);
                                        }),
                                    "New solution on " + std::to_string(idx)));

                if (maybe_exception.has_value())
                    std::rethrow_exception(*maybe_exception);
            }),
        sim->simulator->now() + ts.t,
        "Evaluate solution on " + std::to_string(idx));
}
void SimulatedAsynchronousSimpleGA::evaluate_initial_solution(size_t idx, Individual &ii)
{
    Population &pop = *population;
    GObjectiveFunction &objective_function = *pop.getGlobalData<GObjectiveFunction>();
    doCache();

    TimeSpent &ts = cache->tgts.getData(ii);
    ts.t = 0;
    std::optional<std::exception_ptr> maybe_exception;
    try
    {
        objective_function.of->evaluate(ii);
    }
    catch (std::exception &e)
    {
        maybe_exception = std::current_exception();
    }

    // Wait until a processor is available.
    sim->simulator->simulate_until([this]() { return sim->num_workers_busy < sim->num_workers; });

    ++sim->num_workers_busy;
    sim->simulator->insert_event(
        std::make_unique<FunctionalSimResumable>(
            [idx, ii, maybe_exception, this](ISimulator &, double, std::unique_ptr<IResumableSimulated> &) {
                // Perform replacement -- initial solutions for async should specifically replace their original index.
                // Hence the override here.
                place_in_population(idx, ii, 1);
                // Process has completed
                --sim->num_workers_busy;
                // Queue up next replacement for this index.
                sim->processing_queue.push(
                    std::make_tuple(0.0,
                                    std::make_unique<FunctionalSimResumable>(
                                        [this, idx](ISimulator &, double, std::unique_ptr<IResumableSimulated> &) {
                                            sample_and_evaluate_new_solution(idx);
                                        }),
                                    "New solution on " + std::to_string(idx)));

                if (maybe_exception.has_value())
                    std::rethrow_exception(*maybe_exception);
            }),
        sim->simulator->now() + ts.t,
        "Evaluate initial solution " + std::to_string(idx));
}
SimulatedAsynchronousSimpleGA::SimulatedAsynchronousSimpleGA(
    std::shared_ptr<SimulatorParameters> sim,
    size_t population_size,
    size_t offspring_size,
    std::shared_ptr<ISolutionInitializer> initializer,
    std::shared_ptr<ICrossover> crossover,
    std::shared_ptr<IMutation> mutation,
    std::shared_ptr<ISelection> parent_selection,
    std::shared_ptr<IPerformanceCriterion> performance_criterion,
    std::shared_ptr<IArchive> archive,
    bool include_population,
    std::shared_ptr<ISelection> generationalish_selection,
    std::shared_ptr<SpanLogger> sl) :
    SimulatedSynchronousSimpleGA(sim,
                                 population_size,
                                 offspring_size,
                                 initializer,
                                 crossover,
                                 mutation,
                                 parent_selection,
                                 performance_criterion,
                                 archive,
                                 include_population,
                                 generationalish_selection,
                                 sl)
{
}
