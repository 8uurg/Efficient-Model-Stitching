#include "d-ga.hpp"
#include "archive.hpp"
#include "base.hpp"
#include "cppassert.h"
#include "decentralized.hpp"
#include "scalarize.hpp"
#include <random>

void schedule_init_generation_ga(
    std::shared_ptr<Scheduler> &wd,
    Population &population,
    std::vector<Individual> &individuals,
    size_t population_size,
    ISolutionInitializer *initializer,
    IArchive *archive,
    std::optional<TypedGetter<PopulationIndex>> maybe_tgpi)
{
    std::unique_ptr<WaitForOtherEventsResumable> wfoer(new WaitForOtherEventsResumable(wd->tag_stack.back()));
    WaitForOtherEventsResumable *wfoer_ = wfoer.get();
    wd->tag_stack.pop_back();
    size_t completion_tag = wd->get_tag(std::move(wfoer));

    individuals.resize(population_size);

    for (size_t i = 0; i < population_size; ++i)
        individuals[i] = population.newIndividual();

    initializer->initialize(individuals);

    if (maybe_tgpi.has_value())
    {
        auto &tgpi = *maybe_tgpi;
        for (size_t i = 0; i < population_size; ++i)
        {
            tgpi.getData(individuals[i]).idx = i;
        }
    }

    GObjectiveFunction &objective_function = *population.getGlobalData<GObjectiveFunction>();

    for (auto ii : individuals)
    {
        // Note: Unlike the simulator, some tasks simply have a completion time farther into the future
        //       if the resources are all busy. No need to keep track of active # of processors.
        // Note: Nor do we need to spend resources on managing time.
        // We do however need to keep track of how many are pending, so we know when everything has completed.
        wfoer_->wait_for_one();

        // Note: Evaluation may throw an exception prior to, and after evaluation.
        //  But: not sure how to handle cases where it is prior to where it goes wrong.

        // Register a handler for when function evaluation completes.
        auto tag = wd->get_tag(std::make_unique<FunctionalResumable>([archive, ii, completion_tag](Scheduler &wd) {
            // When evaluation completes...

            // Add solution to archive
            archive->try_add(ii);

            // Inform that an evaluation task has been completed
            wd.complete_tag(completion_tag);

            return false;
        }));

        wd->tag_stack.push_back(tag);

        // Request function evaluation.
        try {
            objective_function.of->evaluate(ii);
        } catch (run_complete &e) {
            // See (note:evalex)
        }
    }

    // Unlike the simulator, this function does not terminate once initialization finishes.
    // Make sure to register a Resumable using the tag-stack!
}


// DistributedSynchronousSimpleGA

DistributedSynchronousSimpleGA::DistributedSynchronousSimpleGA(std::shared_ptr<Scheduler> sp,
                                                           size_t population_size,
                                                           size_t offspring_size,
                                                           int replacement_strategy,
                                                           std::shared_ptr<ISolutionInitializer> initializer,
                                                           std::shared_ptr<ICrossover> crossover,
                                                           std::shared_ptr<IMutation> mutation,
                                                           std::shared_ptr<ISelection> parent_selection,
                                                           std::shared_ptr<IPerformanceCriterion> performance_criterion,
                                                           std::shared_ptr<IArchive> archive) :
    sp(sp),
    initializer(initializer),
    crossover(crossover),
    mutation(mutation),
    parent_selection(parent_selection),
    performance_criterion(performance_criterion),
    archive(archive),
    population_size(population_size),
    offspring_size(offspring_size),
    replacement_strategy(replacement_strategy)
{
}
DistributedSynchronousSimpleGA::DistributedSynchronousSimpleGA(std::shared_ptr<Scheduler> sp,
                                                           size_t population_size,
                                                           size_t offspring_size,
                                                           std::shared_ptr<ISolutionInitializer> initializer,
                                                           std::shared_ptr<ICrossover> crossover,
                                                           std::shared_ptr<IMutation> mutation,
                                                           std::shared_ptr<ISelection> parent_selection,
                                                           std::shared_ptr<IPerformanceCriterion> performance_criterion,
                                                           std::shared_ptr<IArchive> archive,
                                                           bool include_population,
                                                           std::shared_ptr<ISelection> generationalish_selection) :
    sp(sp),
    initializer(initializer),
    crossover(crossover),
    mutation(mutation),
    parent_selection(parent_selection),
    performance_criterion(performance_criterion),
    archive(archive),
    population_size(population_size),
    offspring_size(offspring_size),
    replacement_strategy(6),
    target_selection_pool_size(offspring_size),
    include_population(include_population),
    generationalish_selection(generationalish_selection)
{
}

void DistributedSynchronousSimpleGA::initialize()
{
    doCache();

    // After completing initialization, the next step is to start the next generation.
    auto tag = sp->get_tag(std::make_unique<FunctionalResumable>([this](Scheduler & /* wd */) {
        this->generation();
        return false;
    }));
    sp->tag_stack.push_back(tag);
    
    schedule_init_generation_ga(
        sp,
        *population,
        individuals,
        population_size,
        initializer.get(),
        archive.get(),
        // Note: set population indices if we have cached the getter (as this means we require it.)
        cache->population_idx
    );
    offspring.resize(offspring_size);
    population->newIndividuals(offspring);
}
void DistributedSynchronousSimpleGA::replace_fifo(Individual ii)
{
    Population &pop = *population;
    pop.copyIndividual(ii, individuals[current_replacement_index]);
    current_replacement_index += 1;
    if (current_replacement_index >= population_size)
        current_replacement_index = 0;
}
void DistributedSynchronousSimpleGA::replace_idx(size_t idx, Individual ii)
{
    Population &pop = *population;
    pop.copyIndividual(ii, individuals[idx]);
}
void DistributedSynchronousSimpleGA::replace_uniformly(Individual ii)
{
    Population &pop = *population;
    std::uniform_int_distribution<size_t> d_pop_idx(0, population_size - 1);
    size_t idx = d_pop_idx(cache->rng.rng);
    pop.copyIndividual(ii, individuals[idx]);
}
void DistributedSynchronousSimpleGA::replace_selection_fifo(Individual ii)
{
    Population &pop = *population;
    size_t idx = current_replacement_index++;
    short c = performance_criterion->compare(individuals[idx], ii);
    if ((c == 3 && replace_if_equal) || (c == 0 && replace_if_incomparable) || (c == 2))
    {
        pop.copyIndividual(ii, individuals[idx]);
    }
    if (current_replacement_index >= population_size)
        current_replacement_index = 0;
}
void DistributedSynchronousSimpleGA::replace_selection_idx(size_t idx, Individual ii)
{
    Population &pop = *population;
    short c = performance_criterion->compare(individuals[idx], ii);
    if ((c == 3 && replace_if_equal) || (c == 0 && replace_if_incomparable) || (c == 2))
    {
        pop.copyIndividual(ii, individuals[idx]);
    }
}
void DistributedSynchronousSimpleGA::replace_one_idx(Individual ii, std::vector<size_t> &idxs)
{
    Population &pop = *population;
    Rng &rng = *pop.getGlobalData<Rng>();
    auto &gpopidx = *cache->population_idx;
    std::optional<Individual> parent_to_replace;
    size_t num_before = 0;
    size_t parent_idx = 0;

    for (auto idx: idxs)
    {
        short c = performance_criterion->compare(individuals[idx], ii);
        if ((c == 3 && replace_if_equal) || (c == 0 && replace_if_incomparable) || (c == 2))
        {
            bool replace = false;
            if (num_before == 0)
            {
                replace = true;
            }
            else
            {
                // With a probability of 1 / (num_before + 1) replace.
                std::uniform_int_distribution<size_t> do_we_replace(0, num_before);
                replace = num_before == do_we_replace(rng.rng);
            }
            if (replace)
            {
                parent_to_replace = individuals[idx];
                parent_idx = idx;
            }
        }
    }

    if (parent_to_replace.has_value())
    {
        
        pop.copyIndividual(ii, *parent_to_replace);
        gpopidx.getData(*parent_to_replace).idx = parent_idx;
    }
}

void DistributedSynchronousSimpleGA::familial_replacement(Individual ii)
{
    doCache();
    auto &gpi = *cache->parent_indices;
    auto &pi = gpi.getData(ii);
    t_assert(pi.parent_indices.size() > 0, "parent indices were left unset.");
    replace_one_idx(ii, pi.parent_indices);
}
void DistributedSynchronousSimpleGA::population_replacement(Individual ii)
{
    doCache();
    auto &gpi = *cache->parent_indices;
    auto &pi = gpi.getData(ii);
    t_assert(pi.parent_indices.size() > 0, "parent indices were left unset.");
    replace_one_idx(ii, pi.parent_indices);
}
void DistributedSynchronousSimpleGA::replace_selection_uniformly(Individual ii)
{
    Population &pop = *population;
    std::uniform_int_distribution<size_t> d_pop_idx(0, population_size - 1);
    size_t idx = d_pop_idx(cache->rng.rng);
    short c = performance_criterion->compare(individuals[idx], ii);
    if ((c == 3 && replace_if_equal) || (c == 0 && replace_if_incomparable) || (c == 2))
    {
        pop.copyIndividual(ii, individuals[idx]);
    }
}
void DistributedSynchronousSimpleGA::contender_generational_like_selection(Individual ii)
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
void DistributedSynchronousSimpleGA::place_in_population(size_t idx,
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
    case 7:
        // Familial replacement - replace one of the parents against which you are better.
        familial_replacement(ii);
        break;
    case 8:
        // Population replacement - replace a solution from the population against which you are better.
        population_replacement(ii);
        break;
    }
}
void DistributedSynchronousSimpleGA::evaluate_solution(size_t idx, Individual ii)
{
    Population &pop = *population;
    GObjectiveFunction &objective_function = *pop.getGlobalData<GObjectiveFunction>();

    size_t completion_tag = sp->tag_stack.back();
    sp->tag_stack.pop_back();

    std::optional<std::exception_ptr> maybe_exception;
    auto tag = sp->get_tag(std::make_unique<FunctionalResumable>([this, idx, ii, completion_tag](Scheduler & /* wd */) {
        place_in_population(idx, ii, std::nullopt);
        sp->complete_tag(completion_tag);
        return false;
    }));
    sp->tag_stack.push_back(tag);
    try
    {
        objective_function.of->evaluate(ii);
    }
    catch (run_complete &e)
    {
        maybe_exception = std::current_exception();
    }
}
bool DistributedSynchronousSimpleGA::terminated()
{
    // Not converged if we haven't started yet.
    if (!initialized)
        return false;
    
    // If there are no more actions to take (e.g. due to evaluation budget being
    // hit) - we have terminated too.
    if (sp->terminated())
        return true;

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
void DistributedSynchronousSimpleGA::generation()
{
    if (terminated())
        throw stop_approach();

    // Assign scalarization vectors if necessary.
    assign_scalarization_weights_if_necessary(*population, individuals);

    // Much like the approach above - calling generation does not block until the entire generation has ran.
    // First - after this generation has completed - the next step is to run another generation.
    auto tag = sp->get_tag(std::make_unique<FunctionalResumable>([this](Scheduler & /* wd */) {
        this->generation();
        return false;
    }));
    sp->tag_stack.push_back(tag);

    // This is done after all evaluations have reportedly been completed.
    std::unique_ptr<WaitForOtherEventsResumable> wfoer(new WaitForOtherEventsResumable(sp->tag_stack.back()));
    WaitForOtherEventsResumable *wfoer_ = wfoer.get();
    sp->tag_stack.pop_back();
    size_t completion_tag = sp->get_tag(std::move(wfoer));


    for (size_t idx = 0; idx < offspring.size(); ++idx)
    {
        wfoer_->wait_for_one();
        // In a generational GA the evaluate solution is required to provide feedback when a solution completes evaluation.
        sp->tag_stack.push_back(completion_tag);
        auto &nii = offspring[idx];
        sample_solution(idx, nii);
        evaluate_solution(idx, nii);
    }

}
void DistributedSynchronousSimpleGA::step()
{
    if (!initialized)
    {
        initialize();
        initialized = true;
    }
    else
    {
        sp->step();
    }
}
std::vector<Individual> DistributedSynchronousSimpleGA::sample_solutions()
{
    size_t num_parents = crossover->num_parents();
    std::vector<Individual> parents = parent_selection->select(individuals, num_parents);
    auto c_offspring = crossover->crossover(parents);
    mutation->mutate(c_offspring);

    if (cache->parent_indices.has_value() && cache->population_idx.has_value())
    {
        auto &tgpi = *cache->parent_indices;
        auto &tgpopidx = *cache->population_idx;
        std::vector<size_t> parent_idxs;
        // Collect parent indices.
        for (auto &p: parents)
        {
            parent_idxs.push_back(tgpopidx.getData(p).idx);
        }
        // Assign parent indices
        for (auto &o: c_offspring)
        {
            tgpi.getData(o).parent_indices = parent_idxs;
        }
    }

    return c_offspring;
}
std::vector<Individual> &DistributedSynchronousSimpleGA::getSolutionPopulation()
{
    return individuals;
}
void DistributedSynchronousSimpleGA::setPopulation(std::shared_ptr<Population> population)
{
    GenerationalApproach::setPopulation(population);
    cache.reset();
    this->sp->setPopulation(population);

    this->initializer->setPopulation(population);
    this->crossover->setPopulation(population);
    this->mutation->setPopulation(population);
    this->parent_selection->setPopulation(population);
    this->performance_criterion->setPopulation(population);
    this->archive->setPopulation(population);
    if (generationalish_selection != NULL)
        this->generationalish_selection->setPopulation(population);
}
void DistributedSynchronousSimpleGA::registerData()
{
    GenerationalApproach::registerData();
    this->sp->registerData();

    if (replacement_strategy == 7)
    {
        // We need to keep track of things with this strategy. :)
        population->registerData<PopulationIndex>();
        population->registerData<ParentIndices>();
    }

    this->initializer->registerData();
    this->crossover->registerData();
    this->mutation->registerData();
    this->parent_selection->registerData();
    this->performance_criterion->registerData();
    this->archive->registerData();
    if (generationalish_selection != NULL)
        this->generationalish_selection->registerData();

    // If there are scalarizers registered - we will also generate scalarization weights.
    if (population->isGlobalRegistered<ScalarizerIndex>()) {
        population->registerData<ScalarizationWeights>();
    }
}
void DistributedSynchronousSimpleGA::afterRegisterData()
{
    GenerationalApproach::afterRegisterData();
    this->sp->afterRegisterData();

    this->initializer->afterRegisterData();
    this->crossover->afterRegisterData();
    this->mutation->afterRegisterData();
    this->parent_selection->afterRegisterData();
    this->performance_criterion->afterRegisterData();
    this->archive->afterRegisterData();
    if (generationalish_selection != NULL)
        this->generationalish_selection->afterRegisterData();
}

// DistributedAsynchronousSimpleGA

DistributedAsynchronousSimpleGA::DistributedAsynchronousSimpleGA(
    std::shared_ptr<Scheduler> sp,
    size_t population_size,
    size_t offspring_size,
    int replacement_strategy,
    std::shared_ptr<ISolutionInitializer> initializer,
    std::shared_ptr<ICrossover> crossover,
    std::shared_ptr<IMutation> mutation,
    std::shared_ptr<ISelection> parent_selection,
    std::shared_ptr<IPerformanceCriterion> performance_criterion,
    std::shared_ptr<IArchive> archive) :
    DistributedSynchronousSimpleGA(sp,
                                 population_size,
                                 offspring_size,
                                 replacement_strategy,
                                 initializer,
                                 crossover,
                                 mutation,
                                 parent_selection,
                                 performance_criterion,
                                 archive)
{
}
void DistributedAsynchronousSimpleGA::initialize()
{
    Population &population = *this->population;

    individuals.resize(population_size);
    population.newIndividuals(individuals);
    initializer->initialize(individuals);

    offspring.resize(offspring_size);
    population.newIndividuals(offspring);

    doCache();
    if (cache->population_idx.has_value())
    {
        // Assign population indices.
        auto &tgpi = *cache->population_idx;
        for (size_t i = 0; i < population_size; ++i)
        {
            tgpi.getData(individuals[i]).idx = i;
        }
    }

    for (size_t idx = 0; idx < individuals.size(); ++idx)
    {
        population.copyIndividual(individuals[idx], offspring[idx]);
        // Initial evaluations should always greedily replace the original individual
        // to simulate the actual completion of evaluation of this solution.
        evaluate_initial_solution(idx, offspring[idx]);
    }

    // sim->simulator->simulate_until_end();
}
void DistributedAsynchronousSimpleGA::generation()
{
    if (terminated())
        throw stop_approach();

    for (size_t num = 0; num < population_size; ++num)
    {
        if (sp->terminated()) return;
        sp->step();
    }
}
void DistributedAsynchronousSimpleGA::sample_and_evaluate_new_solution(size_t idx)
{
    Individual ii = offspring[idx];
    sample_solution(idx, ii);
    evaluate_solution(idx, ii);
}
void DistributedAsynchronousSimpleGA::evaluate_solution(size_t idx, Individual ii)
{
    Population &pop = *population;
    GObjectiveFunction &objective_function = *pop.getGlobalData<GObjectiveFunction>();

    std::optional<std::exception_ptr> maybe_exception;
    auto tag = sp->get_tag(std::make_unique<FunctionalResumable>([this, idx, ii](Scheduler & /* wd */) {
        // Unlike the synchronoous implementation, this implementation does not report back.
        // Assign weights prior to placing in population (if necessary)
        if (steps_left_until_weight_reassignment == 0)
        {
            // Assign scalarization vectors if necessary.
            assign_scalarization_weights_if_necessary(*population, individuals);
            steps_left_until_weight_reassignment = population_size;
        }
        else
        {
            steps_left_until_weight_reassignment -= 1;
        }
        // It does try to insert the solution into the population
        place_in_population(idx, ii, std::nullopt);
        // And then request the next sampling & evaluation directly (rather than asking a generational clock to do so.)
        sp->schedule_immediately(
            std::make_unique<FunctionalResumable>(
            [this, idx](Scheduler &) {
                sample_and_evaluate_new_solution(idx);
                return false;
            })
            , 0.0);
        return false;
    }));
    sp->tag_stack.push_back(tag);
    try
    {
        objective_function.of->evaluate(ii);
    }
    catch (run_complete &e)
    {
        maybe_exception = std::current_exception();
    }
}
void DistributedAsynchronousSimpleGA::evaluate_initial_solution(size_t idx, Individual &ii)
{
    Population &pop = *population;
    GObjectiveFunction &objective_function = *pop.getGlobalData<GObjectiveFunction>();
    doCache();

    auto evaluation_complete_tag = sp->get_tag(std::make_unique<FunctionalResumable>(
            [idx, ii, this](Scheduler &) {
                // Perform replacement -- initial solutions for async should specifically replace their original index.
                // Hence the override here.
                place_in_population(idx, ii, 1);
                // Queue up next replacement for this index.
                sp->schedule_immediately(std::make_unique<FunctionalResumable>(
                                        [this, idx](Scheduler &) {
                                            sample_and_evaluate_new_solution(idx);
                                            return false;
                                        }), 0.0);

                return false;
            }));
    sp->tag_stack.push_back(evaluation_complete_tag);

    try
    {
        objective_function.of->evaluate(ii);
    }
    catch (std::exception &e)
    {
    }
}
DistributedAsynchronousSimpleGA::DistributedAsynchronousSimpleGA(
    std::shared_ptr<Scheduler> sp,
    size_t population_size,
    size_t offspring_size,
    std::shared_ptr<ISolutionInitializer> initializer,
    std::shared_ptr<ICrossover> crossover,
    std::shared_ptr<IMutation> mutation,
    std::shared_ptr<ISelection> parent_selection,
    std::shared_ptr<IPerformanceCriterion> performance_criterion,
    std::shared_ptr<IArchive> archive,
    bool include_population,
    std::shared_ptr<ISelection> generationalish_selection) :
    DistributedSynchronousSimpleGA(sp,
                                 population_size,
                                 offspring_size,
                                 initializer,
                                 crossover,
                                 mutation,
                                 parent_selection,
                                 performance_criterion,
                                 archive,
                                 include_population,
                                 generationalish_selection)
{
}
bool DistributedAsynchronousSimpleGA::terminated()
{
    return initialized && sp->terminated();
}

DistributedRandomSearch::DistributedRandomSearch(std::shared_ptr<Scheduler> sp,
                                                 std::shared_ptr<ISolutionInitializer> initializer,
                                                 std::shared_ptr<IArchive> archive,
                                                 size_t max_pending) :
    sp(sp), initializer(initializer), archive(archive), max_pending(max_pending)
{
}
bool DistributedRandomSearch::terminated()
{
    return initialized && sp->terminated();
};
void DistributedRandomSearch::evaluate_solution_at(size_t idx)
{
    // Evaluate -> schedule replacement & next evaluation.
    Population &pop = *population;
    Individual ii = pending[idx];
    GObjectiveFunction &objective_function = *pop.getGlobalData<GObjectiveFunction>();

    auto evaluation_complete_tag = sp->get_tag(std::make_unique<FunctionalResumable>([idx, ii, this](Scheduler &) {
        archive->try_add(ii);
        // Queue up next replacement for this index.
        sp->schedule_immediately(std::make_unique<FunctionalResumable>([this, idx](Scheduler &) {
                                     sample_and_evaluate(idx);
                                     return false;
                                 }),
                                 0.0);

        return false;
    }));
    sp->tag_stack.push_back(evaluation_complete_tag);

    try
    {
        objective_function.of->evaluate(ii);
    }
    catch (std::exception &e)
    {
    }
}
void DistributedRandomSearch::sample_and_evaluate(size_t idx)
{
    Individual ii = pending[idx];
    if (batch_idx >= next_batch.size())
    {
        initializer->initialize(next_batch);
        batch_idx = 0;
    }
    population->copyIndividual(next_batch[batch_idx], ii);
    batch_idx += 1;
    evaluate_solution_at(idx);
}
void DistributedRandomSearch::initialize()
{
    pending.resize(max_pending);
    next_batch.resize(max_pending);
    population->newIndividuals(pending);
    population->newIndividuals(next_batch);
    initializer->initialize(pending);
    initializer->initialize(next_batch);

    for (size_t i = 0; i < max_pending; i++)
    {
        evaluate_solution_at(i);
    }

    initialized = true;
}
void DistributedRandomSearch::step()
{
    if (!initialized)
    {
        initialize();
    }
    else
    {
        sp->step();
    }
}
void DistributedRandomSearch::setPopulation(std::shared_ptr<Population> population)
{
    GenerationalApproach::setPopulation(population);
    sp->setPopulation(population);
    initializer->setPopulation(population);
    archive->setPopulation(population);
}
void DistributedRandomSearch::registerData()
{
    GenerationalApproach::registerData();
    sp->registerData();
    archive->registerData();
}
void DistributedRandomSearch::afterRegisterData()
{
    GenerationalApproach::afterRegisterData();
    sp->afterRegisterData();
    archive->afterRegisterData();
}
std::vector<Individual> &DistributedRandomSearch::getSolutionPopulation()
{
    // Note - not really useful...
    return pending;
}
