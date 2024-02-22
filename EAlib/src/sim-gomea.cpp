#include <exception>
#include <memory>
#include <optional>

#include "base.hpp"
#include "cppassert.h"
#include "gomea.hpp"
#include "sim-gomea.hpp"
#include "sim.hpp"

// -- Reused Generational (Non-Simulated) -> Generational (Simulated)

/**
 * @brief Simulate the parallel initialization of GOMEA.
 *
 * @param sp Base simulator to use.
 * @param base Approach to initialize the population for.
 */
void simulate_init_generation(std::shared_ptr<SimulatorParameters> &sp, std::shared_ptr<SpanLogger> sl, BaseGOMEA *base)
{
    Population &population = *base->population;
    base->individuals.resize(base->population_size);

    for (size_t i = 0; i < base->population_size; ++i)
        base->individuals[i] = population.newIndividual();

    base->initializer->initialize(base->individuals);

    GObjectiveFunction &objective_function = *population.getGlobalData<GObjectiveFunction>();
    TypedGetter<TimeSpent> tgts = population.getDataContainer<TimeSpent>();

    for (auto ii : base->individuals)
    {
        sp->simulator->simulate_until([&sp]() { return sp->num_workers_busy < sp->num_workers; });
        // Note: as there are no interactions for a synchronous algorithm (archive is unused, only initial evaluations
        // are performed), we can simply schedule a event that frees up the processor
        //       again. And perform the evaluation and addition to archive with the current individual immediately.
        ++sp->num_workers_busy;
        auto &d = tgts.getData(ii);
        d.t = 0;
        // Note: function may throw an exception, i.e. vtr found,
        // evaluation limit reached...
        std::optional<std::exception_ptr> maybe_exception;
        try
        {
            objective_function.of->evaluate(ii);
        }
        catch (std::exception &e)
        {
            maybe_exception = std::current_exception();
        }

        sp->simulator->insert_event(
            std::make_unique<FunctionalSimResumable>(
                [&sp, &base, maybe_exception, ii, sl](ISimulator &, double, std::unique_ptr<IResumableSimulated> &) {
                    --sp->num_workers_busy;
                    base->archive->try_add(ii);
                    auto iic = ii;
                    if (sl != NULL)
                    {
                        sl->start_span(ii.i, iic, 0);
                    }

                    if (maybe_exception.has_value())
                        std::rethrow_exception(*maybe_exception);
                }),
            sp->simulator->now() + d.t,
            "Initial evaluation for solution " + std::to_string(ii.i));
    }
    // Ensure that all solutions have had their full evaluation time finished before starting
    // the actual generations. (this is a generational algorithm after all.)
    sp->simulator->simulate_until_end();
}

/**
 * @brief Simulate a parallel generation of GOMEA.
 *
 * @tparam T A lambda/function pointer of type void(Individual, Simulator &)
 * @param sp Simulator to use
 * @param individuals Population of solutions to perform a generation on
 * @param improveSolution the solution specific improvement operator (i.e. GOM iteration)
 */
template <typename T>
void simulate_step_generation(std::shared_ptr<SimulatorParameters> &sp,
                              std::vector<Individual> &individuals,
                              T &&improveSolution)
{
    static_assert(std::is_invocable<T, size_t, Individual, Simulator &>::value,
                  "signature improveSolution is not void(Individual, Simulator &)");

    for (size_t idx = 0; idx < individuals.size(); ++idx)
    {
        auto &ii = individuals[idx];
        // Simulation: Wait until a processor is available.
        // Process the completion of any individuals waiting for a time-skip, if needed.
        sp->simulator->simulate_until([&sp]() { return (sp->num_workers_busy < sp->num_workers); });
        t_assert(sp->num_workers_busy < sp->num_workers,
                 "No more tasks, yet simulated workers are still busy. This is a bug.");
        // Schedule an improvement attempt.
        improveSolution(idx, ii, *sp->simulator);
    }
    sp->simulator->simulate_until_end();
}

// -- State Machines --

// StepwiseSimulatedGOM
SimulatedGOM::SimulatedGOM(Population &population,
                           Individual to_improve,
                           APtr<FoS> fos,
                           APtr<ISamplingDistribution> distribution,
                           APtr<IPerformanceCriterion> acceptance_criterion,
                           std::function<void(Individual &, Individual &)> replace_in_population,
                           bool *changed,
                           bool *improved) :
    population(population),
    to_improve(to_improve),
    backup(population.newIndividual()),
    current(population.newIndividual()),
    fos(std::move(fos)),
    distribution(std::move(distribution)),
    acceptance_criterion(std::move(acceptance_criterion)),
    replace_in_population(replace_in_population),
    changed(changed),
    improved(improved)
{
    population.copyIndividual(to_improve, *backup);
    population.copyIndividual(to_improve, *current);
    *changed = false;
    *improved = false;
}
std::unique_ptr<SimulatedGOM> SimulatedGOM::apply(Population &population,
                                                  Individual to_improve,
                                                  APtr<FoS> fos,
                                                  APtr<ISamplingDistribution> distribution,
                                                  APtr<IPerformanceCriterion> acceptance_criterion,
                                                  std::function<void(Individual &, Individual &)> replace_in_population,
                                                  bool *changed,
                                                  bool *improved)
{
    return std::make_unique<SimulatedGOM>(population,
                                          to_improve,
                                          std::move(fos),
                                          std::move(distribution),
                                          std::move(acceptance_criterion),
                                          std::move(replace_in_population),
                                          changed,
                                          improved);
}

void SimulatedGOM::evaluate_change(Individual current,
                                   Individual /* backup */,
                                   std::vector<size_t> & /* elements_changed */)
{
    //
    GObjectiveFunction &objective_function = *population.getGlobalData<GObjectiveFunction>();
    objective_function.of->evaluate(current);

    // TODO: Add partial evaluations here when supported!
}
void SimulatedGOM::resume(ISimulator &simulator, double at, std::unique_ptr<IResumableSimulated> &self)
{
    auto fosp = as_ptr(fos);
    if (idx > fosp->size())
    {
        return;
    }
    if (idx > 0)
    {
        // Finish up previous evaluation.
        short performance_judgement = as_ptr(acceptance_criterion)->compare(*backup, *current);
        if (performance_judgement == 1)
        {
            // Backup is better than current, change made the solution worse, revert.
            population.copyIndividual(*backup, *current);
        }
        else
        {
            // Solution is improved by change. Update backup.
            population.copyIndividual(*current, *backup);
            if (changed != nullptr)
            {
                *changed = true;
            }
            if (performance_judgement == 2 && improved != nullptr)
                *improved = true;
        }
        // TODO:
        //  Update of population can also be made more often!
        //  Updating only after a full iteration of GOM is a design decision
        //  that needs to be analysed.
        if (idx == fosp->size() - 1)
        {
            // Update actual population
            // population.copyIndividual(*current, to_improve);
            replace_in_population(*current, to_improve);
        }

        if (maybe_exception.has_value())
        {
            std::rethrow_exception(*maybe_exception);
        }
    }
    if (idx < fosp->size())
    {
        // Set up next evaluation
        for (; idx < fosp->size(); ++idx)
        {
            FoSElement &e = (*fosp)[idx];
            bool sampling_changed = as_ptr(distribution)->apply_resample(*current, e);
            if (!sampling_changed)
            {
                continue;
            }

            TimeSpent &ts_c = population.getData<TimeSpent>(*current);
            ts_c.t = 0;

            // Evaluate change
            try
            {
                evaluate_change(*current, *backup, e);
            }
            catch (std::exception &e)
            {
                maybe_exception = std::current_exception();
            }

            // Simulate cost of evaluation.
            double event_time = at + ts_c.t;
            // Next step!
            ++idx;
            // Add to simulator
            simulator.insert_event(std::move(self),
                                   event_time,
                                   "GOM on " + std::to_string(to_improve.i) + ": evaluating change for fos element " +
                                       std::to_string(idx));
            break;
        }
    }
    return;
}

SimulatedFI::SimulatedFI(Population &population,
                         Individual to_improve,
                         APtr<FoS> fos,
                         APtr<ISamplingDistribution> distribution,
                         APtr<IPerformanceCriterion> acceptance_criterion,
                         std::function<void(Individual &, Individual &)> replace_in_population,
                         bool *changed,
                         bool *improved) :
    population(population),
    to_improve(to_improve),
    backup(population.newIndividual()),
    current(population.newIndividual()),
    fos(std::move(fos)),
    distribution(std::move(distribution)),
    acceptance_criterion(std::move(acceptance_criterion)),
    replace_in_population(replace_in_population),
    changed(changed),
    improved(improved)
{
    population.copyIndividual(to_improve, *backup);
    population.copyIndividual(to_improve, *current);
    *changed = false;
    *improved = false;
}
std::unique_ptr<SimulatedFI> SimulatedFI::apply(Population &population,
                                                Individual to_improve,
                                                APtr<FoS> fos,
                                                APtr<ISamplingDistribution> distribution,
                                                APtr<IPerformanceCriterion> acceptance_criterion,
                                                std::function<void(Individual &, Individual &)> replace_in_population,
                                                bool *changed,
                                                bool *improved)
{
    return std::make_unique<SimulatedFI>(population,
                                         to_improve,
                                         std::move(fos),
                                         std::move(distribution),
                                         std::move(acceptance_criterion),
                                         std::move(replace_in_population),
                                         changed,
                                         improved);
}

void SimulatedFI::evaluate_change(Individual current,
                                  Individual /* backup */,
                                  std::vector<size_t> & /* elements_changed */)
{
    //
    GObjectiveFunction &objective_function = *population.getGlobalData<GObjectiveFunction>();
    objective_function.of->evaluate(current);

    // TODO: Add partial evaluations here when supported!
}
void SimulatedFI::resume(ISimulator &simulator, double at, std::unique_ptr<IResumableSimulated> &self)
{
    auto fosp = as_ptr(fos);
    if (idx > fosp->size() || success)
    {
        return;
    }
    if (idx > 0)
    {
        // Finish up previous evaluation.
        short performance_judgement = as_ptr(acceptance_criterion)->compare(*backup, *current);
        if (performance_judgement == 1)
        {
            // Backup is better than current, change made the solution worse, revert.
            population.copyIndividual(*backup, *current);
        }
        else
        {
            // Update actual population
            // population.copyIndividual(*current, to_improve);
            replace_in_population(*current, to_improve);
            success = true;

            if (changed != nullptr)
            {
                *changed = true;
            }
            if (performance_judgement == 2 && improved != nullptr)
                *improved = true;
        }
        if (maybe_exception.has_value())
        {
            std::rethrow_exception(*maybe_exception);
        }
    }
    if (idx < fosp->size() && !success)
    {
        // Set up next evaluation
        for (; idx < fosp->size(); ++idx)
        {
            FoSElement &e = (*fosp)[idx];
            bool sampling_changed = as_ptr(distribution)->apply_resample(*current, e);
            if (!sampling_changed)
            {
                continue;
            }

            TimeSpent &ts_c = population.getData<TimeSpent>(*current);
            ts_c.t = 0;

            // Evaluate change
            try
            {
                evaluate_change(*current, *backup, e);
            }
            catch (std::exception &e)
            {
                maybe_exception = std::current_exception();
            }

            // Simulate cost of evaluation.
            double event_time = at + ts_c.t;
            // Next fos element - maybe
            ++idx;
            // Add to simulator
            simulator.insert_event(std::move(self),
                                   event_time,
                                   "FI on " + std::to_string(to_improve.i) + ": evaluating change for fos element " +
                                       std::to_string(idx));
            break;
        }
    }
    return;
}

// SimulatedGOMThenMaybeFI
SimulatedGOMThenMaybeFI::SimulatedGOMThenMaybeFI(Population &population,
                                                 Individual to_improve,
                                                 std::function<size_t(Individual &ii)> getNISThreshold,
                                                 std::function<Individual(Individual &ii)> getReplacementSolution,
                                                 std::function<void(Individual &, Individual &)> replace_in_population,
                                                 std::function<void(ISimulator &simulator, double)> onCompletion,
                                                 std::unique_ptr<SimIGOMFIData> kernel_data) :
    population(population),
    to_improve(to_improve),
    kernel_data(std::move(kernel_data)),
    getNISThreshold(getNISThreshold),
    getReplacementSolution(getReplacementSolution),
    replace_in_population(replace_in_population),
    onCompletion(onCompletion)
{
    improved = std::make_unique<bool>(false);
    changed = std::make_unique<bool>(false);
    auto gom = SimulatedGOM::apply(population,
                                   to_improve,
                                   this->kernel_data->getFOSForGOM(),
                                   this->kernel_data->getDistributionForGOM(),
                                   this->kernel_data->getPerformanceCriterionForGOM(),
                                   this->replace_in_population,
                                   changed.get(),
                                   improved.get());
    enqueued.push(std::make_tuple(0.0, std::move(gom), "Applying GOM on solution " + std::to_string(to_improve.i)));
}
void SimulatedGOMThenMaybeFI::resume(ISimulator &simulator, double at, std::unique_ptr<IResumableSimulated> &self)
{
    // If there is a currently active suboperation: forward.
    if (enqueued.size() > 0)
    {
        auto &front = enqueued.front();
        std::get<1>(front)->resume(*this, at, std::get<1>(front));
        enqueued.pop();
    }
    // If this operation did not enqueue anything new: it is done!
    // Lookup what the next point of action should be.
    // Perform FI?
    if (enqueued.size() == 0)
    {
        switch (state)
        {
        case GOM: {
            NIS &nis = population.getData<NIS>(to_improve);
            if (!improved)
                nis.nis += 1;

            if (improved)
            {
                onImprovedSolution(to_improve);
            }

            // If solution hasn't changed, or the NIS threshold has been reached
            // perform Forced Improvements
            if (!changed || nis.nis > getNISThreshold(to_improve))
            {
                auto fi = SimulatedFI::apply(population,
                                             to_improve,
                                             kernel_data->getFOSForFI(),
                                             kernel_data->getDistributionForFI(),
                                             kernel_data->getPerformanceCriterionForFI(),
                                             this->replace_in_population,
                                             changed.get(),
                                             improved.get());
                state = FI;
                enqueued.push(
                    std::make_tuple(at, std::move(fi), "Applying FI on solution " + std::to_string(to_improve.i)));
            }
            else
            {
                onCompletion(simulator, at);
                state = Completed;
            }
        }
        break;
        case FI:
            if (improved)
            {
                onImprovedSolution(to_improve);
            }
            if (!changed)
            {
                population.copyIndividual(getReplacementSolution(to_improve), to_improve);
            }
            onCompletion(simulator, at);
            state = Completed;
            break;
        case Completed:
            break;
        }
    }
    else
    {
        simulator.insert_event(std::move(self), std::get<0>(enqueued.front()), std::get<2>(enqueued.front()));
    }
}
void SimulatedGOMThenMaybeFI::insert_event(std::unique_ptr<IResumableSimulated> e,
                                           double at,
                                           std::optional<std::string> descriptor)
{
    enqueued.push(std::make_tuple(at, std::move(e), std::move(descriptor)));
}
void SimulatedGOMThenMaybeFI::onImprovedSolution(Individual &ii)
{
    // Reset NIS
    NIS &nis = population.getData<NIS>(ii);
    nis.nis = 0;
}

// SimParallelSynchronousGOMEA
SimParallelSynchronousGOMEA::SimParallelSynchronousGOMEA(std::shared_ptr<SimulatorParameters> sp,
                                                         size_t population_size,
                                                         std::shared_ptr<ISolutionInitializer> initializer,
                                                         std::shared_ptr<FoSLearner> foslearner,
                                                         std::shared_ptr<IPerformanceCriterion> performance_criterion,
                                                         std::shared_ptr<IArchive> archive,
                                                         std::shared_ptr<SpanLogger> sl,
                                                         bool /* donor_search */,
                                                         bool autowrap) :
    GOMEA(population_size, initializer, foslearner, performance_criterion, archive, autowrap),
    sp(std::move(sp)),
    sl(std::move(sl))
{
}
void SimParallelSynchronousGOMEA::setPopulation(std::shared_ptr<Population> population)
{
    GOMEA::setPopulation(population);
    if (sl != NULL)
        sl->setPopulation(population);
}
void SimParallelSynchronousGOMEA::registerData()
{
    GOMEA::registerData();
    if (sl != NULL)
        sl->registerData();

    Population &pop = *this->population;
    pop.registerData<TimeSpent>();
    tgts.emplace(pop.getDataContainer<TimeSpent>());
}
void SimParallelSynchronousGOMEA::afterRegisterData()
{
    GOMEA::afterRegisterData();
    if (sl != NULL)
        sl->afterRegisterData();
}

void SimParallelSynchronousGOMEA::step_normal_generation()
{
    this->atGenerationStart();

    simulate_step_generation(sp, individuals, [this](size_t idx, Individual ii, Simulator &simulator) {
        improveSolution(idx, ii, simulator);
    });

    this->atGenerationEnd();
}
void SimParallelSynchronousGOMEA::initialize()
{
    simulate_init_generation(sp, sl, this);
}
void SimParallelSynchronousGOMEA::improveSolution(size_t idx, Individual ii, Simulator &simulator)
{
    ++sp->num_workers_busy;

    std::unique_ptr<IResumableSimulated> sgommfi = std::make_unique<SimulatedGOMThenMaybeFI>(
        *population,
        ii,
        [this, ii](Individual &) { return getNISThreshold(ii); },
        [this, ii](Individual &) { return getReplacementSolution(ii); },
        [this, idx](Individual &replacement, Individual &in_population) {
            return replace_population_individual(idx, replacement, in_population);
        },
        [this](ISimulator &, double) { --sp->num_workers_busy; },
        std::make_unique<SimGOMFIDataBaseGOMEA>(this, ii));

    sgommfi->resume(simulator, simulator.now(), sgommfi);
}
void SimParallelSynchronousGOMEA::replace_population_individual(size_t idx,
                                                                Individual replacement,
                                                                Individual in_population)
{
    if (sl != NULL)
    {
        sl->end_span(idx, in_population, 0);
        sl->start_span(idx, replacement, 0);
    }
    population->copyIndividual(replacement, in_population);
}

// SimParallelSynchronousMO_GOMEA
SimParallelSynchronousMO_GOMEA::SimParallelSynchronousMO_GOMEA(
    std::shared_ptr<SimulatorParameters> sp,
    size_t population_size,
    size_t number_of_clusters,
    std::vector<size_t> objective_indices,
    std::shared_ptr<ISolutionInitializer> initializer,
    std::shared_ptr<FoSLearner> foslearner,
    std::shared_ptr<IPerformanceCriterion> performance_criterion,
    std::shared_ptr<IArchive> archive,
    std::shared_ptr<GOMEAPlugin> plugin,
    std::shared_ptr<SpanLogger> sl,
    bool donor_search,
    bool autowrap) :
    MO_GOMEA(population_size,
             number_of_clusters,
             objective_indices,
             initializer,
             foslearner,
             performance_criterion,
             archive,
             plugin,
             donor_search,
             autowrap),
    sp(std::move(sp)),
    sl(std::move(sl))
{
}
void SimParallelSynchronousMO_GOMEA::setPopulation(std::shared_ptr<Population> population)
{
    MO_GOMEA::setPopulation(population);
    if (sl != NULL)
        sl->setPopulation(population);
}
void SimParallelSynchronousMO_GOMEA::registerData()
{
    MO_GOMEA::registerData();
    if (sl != NULL)
        sl->registerData();

    Population &pop = *this->population;
    pop.registerData<TimeSpent>();
    tgts.emplace(pop.getDataContainer<TimeSpent>());
}
void SimParallelSynchronousMO_GOMEA::afterRegisterData()
{
    MO_GOMEA::afterRegisterData();
    if (sl != NULL)
        sl->afterRegisterData();
}
void SimParallelSynchronousMO_GOMEA::initialize()
{
    simulate_init_generation(sp, sl, this);
}
void SimParallelSynchronousMO_GOMEA::step_normal_generation()
{
    this->atGenerationStart();

    simulate_step_generation(sp, individuals, [this](size_t idx, Individual ii, Simulator &simulator) {
        improveSolution(idx, ii, simulator);
    });

    this->atGenerationEnd();
}
void SimParallelSynchronousMO_GOMEA::improveSolution(size_t idx, Individual ii, Simulator &simulator)
{
    ++sp->num_workers_busy;

    std::unique_ptr<IResumableSimulated> sgommfi = std::make_unique<SimulatedGOMThenMaybeFI>(
        *population,
        ii,
        [this, ii](Individual &) { return getNISThreshold(ii); },
        [this, ii](Individual &) { return getReplacementSolution(ii); },
        [this, idx](Individual &replacement, Individual &in_population) {
            return replace_population_individual(idx, replacement, in_population);
        },
        [this](ISimulator &, double) { --sp->num_workers_busy; },
        std::make_unique<SimGOMFIDataBaseGOMEA>(this, ii));

    sgommfi->resume(simulator, simulator.now(), sgommfi);
}
void SimParallelSynchronousMO_GOMEA::replace_population_individual(size_t idx,
                                                                   Individual replacement,
                                                                   Individual in_population)
{
    if (sl != NULL)
    {
        sl->end_span(idx, in_population, 0);
        sl->start_span(idx, replacement, 0);
    }
    population->copyIndividual(replacement, in_population);
}

// SimParallelSynchronousKernelGOMEA
SimParallelSynchronousKernelGOMEA::SimParallelSynchronousKernelGOMEA(
    std::shared_ptr<SimulatorParameters> sp,
    size_t population_size,
    size_t number_of_clusters,
    std::vector<size_t> objective_indices,
    std::shared_ptr<ISolutionInitializer> initializer,
    std::shared_ptr<FoSLearner> foslearner,
    std::shared_ptr<IPerformanceCriterion> performance_criterion,
    std::shared_ptr<IArchive> archive,
    std::shared_ptr<GOMEAPlugin> plugin,
    std::shared_ptr<SpanLogger> sl,
    bool donor_search,
    bool autowrap) :
    KernelGOMEA(population_size,
                number_of_clusters,
                objective_indices,
                initializer,
                foslearner,
                performance_criterion,
                archive,
                plugin,
                donor_search,
                autowrap),
    sp(std::move(sp)),
    sl(std::move(sl))
{
}
void SimParallelSynchronousKernelGOMEA::setPopulation(std::shared_ptr<Population> population)
{
    KernelGOMEA::setPopulation(population);
    if (sl != NULL)
        sl->setPopulation(population);
}
void SimParallelSynchronousKernelGOMEA::registerData()
{
    KernelGOMEA::registerData();
    if (sl != NULL)
        sl->registerData();
    Population &pop = *this->population;
    pop.registerData<TimeSpent>();
    tgts.emplace(pop.getDataContainer<TimeSpent>());
}
void SimParallelSynchronousKernelGOMEA::afterRegisterData()
{
    KernelGOMEA::afterRegisterData();
    if (sl != NULL)
        sl->afterRegisterData();
}
void SimParallelSynchronousKernelGOMEA::initialize()
{
    simulate_init_generation(sp, sl, this);
}
void SimParallelSynchronousKernelGOMEA::step_normal_generation()
{
    this->atGenerationStart();

    simulate_step_generation(sp, individuals, [this](size_t idx, Individual ii, Simulator &simulator) {
        improveSolution(idx, ii, simulator);
    });

    this->atGenerationEnd();
}
void SimParallelSynchronousKernelGOMEA::replace_population_individual(size_t idx,
                                                                      Individual replacement,
                                                                      Individual in_population)
{
    if (sl != NULL)
    {
        sl->end_span(idx, in_population, 0);
        sl->start_span(idx, replacement, 0);
    }
    population->copyIndividual(replacement, in_population);
}
void SimParallelSynchronousKernelGOMEA::improveSolution(size_t idx, Individual ii, Simulator &simulator)
{
    ++sp->num_workers_busy;

    std::unique_ptr<IResumableSimulated> sgommfi = std::make_unique<SimulatedGOMThenMaybeFI>(
        *population,
        ii,
        [this, ii](Individual &) { return getNISThreshold(ii); },
        [this, ii](Individual &) { return getReplacementSolution(ii); },
        [this, idx](Individual &replacement, Individual &in_population) {
            return replace_population_individual(idx, replacement, in_population);
        },
        [this](ISimulator &, double) { --sp->num_workers_busy; },
        std::make_unique<SimGOMFIDataBaseGOMEA>(this, ii));

    sgommfi->resume(simulator, simulator.now(), sgommfi);
}

// SimParallelAsynchronousKernelGOMEA
SimParallelAsynchronousKernelGOMEA::SimParallelAsynchronousKernelGOMEA(
    std::shared_ptr<SimulatorParameters> sp,
    size_t population_size,
    size_t number_of_clusters,
    std::vector<size_t> objective_indices,
    std::shared_ptr<ISolutionInitializer> initializer,
    std::shared_ptr<FoSLearner> foslearner,
    std::shared_ptr<IPerformanceCriterion> performance_criterion,
    std::shared_ptr<IArchive> archive,
    // std::shared_ptr<GOMEAPlugin> plugin = std::make_shared<GOMEAPlugin>(),
    std::shared_ptr<SpanLogger> sl,
    bool donor_search,
    bool autowrap) :
    population_size(population_size),
    number_of_clusters(number_of_clusters),
    donor_search(donor_search),
    objective_indices(objective_indices),
    initializer(initializer),
    foslearner(foslearner),
    performance_criterion(
        autowrap ? std::make_shared<ArchiveAcceptanceCriterion>(
                       std::make_shared<WrappedOrSingleSolutionPerformanceCriterion>(performance_criterion), archive)
                 : performance_criterion),
    archive(archive),
    sp(std::move(sp)),
    sl(std::move(sl))
{
}
void SimParallelAsynchronousKernelGOMEA::setPopulation(std::shared_ptr<Population> population)
{
    IDataUser::setPopulation(population);
    initializer->setPopulation(population);
    foslearner->setPopulation(population);
    performance_criterion->setPopulation(population);
    archive->setPopulation(population);
    if (sl != NULL)
        sl->setPopulation(population);
}
void SimParallelAsynchronousKernelGOMEA::registerData()
{
    initializer->registerData();
    foslearner->registerData();
    performance_criterion->registerData();
    archive->registerData();
    if (sl != NULL)
        sl->registerData();

    Population &pop = *this->population;
    pop.registerData<TimeSpent>();
    tgts.emplace(pop.getDataContainer<TimeSpent>());
    pop.registerData<ClusterIndex>();
    pop.registerData<UseSingleObjective>();
    pop.registerData<NIS>();
}
void SimParallelAsynchronousKernelGOMEA::afterRegisterData()
{
    initializer->afterRegisterData();
    foslearner->afterRegisterData();
    performance_criterion->afterRegisterData();
    archive->afterRegisterData();
    if (sl != NULL)
        sl->afterRegisterData();

    Population &population = *this->population;
    rng = population.getGlobalData<Rng>().get();
}
void SimParallelAsynchronousKernelGOMEA::onImprovedSolution(Individual &ii)
{
    Population &population = *this->population;
    // Reset NIS
    NIS &nis = population.getData<NIS>(ii);
    nis.nis = 0;
}

void SimParallelAsynchronousKernelGOMEA::improveSolution(size_t idx, const Individual &ii, Simulator &simulator)
{
    // Note: this is already done in the simulator
    // ++sp->num_workers_busy;

    double t_processing_queued = simulator.now();

    std::unique_ptr<IResumableSimulated> sgommfi = std::make_unique<SimulatedGOMThenMaybeFI>(
        *population,
        ii,
        [this, ii](Individual &) { return getNISThreshold(ii); },
        [this, ii](Individual &) { return getReplacementSolution(ii); },
        [this, idx](Individual &replacement, Individual &in_population) {
            return replace_population_individual(idx, replacement, in_population);
        },
        [this, ii, &simulator, t_processing_queued, idx](ISimulator &, double) {
            --sp->num_workers_busy;
            auto new_resumable = std::make_unique<FunctionalSimResumable>(
                [ii, this, idx](ISimulator &, double, std::unique_ptr<IResumableSimulated> &) {
                    improveSolution(idx, ii, *sp->simulator);
                });
            double t_now = simulator.now();
            if (t_processing_queued == t_now)
            {
                // Time at which we were scheduled to be processed == time of completion.
                // Which means no time has passed (and scheduling it to process again will likely not
                // yield any different results: no time has passed after all)
                // Alternatively, if things have changed, this SHOULD have advanced the clock,
                // this is a bit of an assumption however: if things change without needing
                // evaluations, this could introduce unnecessary waiting time.
                // However: this is frankly not the end of the world, especially compared to time
                // getting stuck indefinitely.
                sp->processing_queue_next_t.push(std::make_tuple(
                    0.0, std::move(new_resumable), "[Timeskip] Start improving solution " + std::to_string(ii.i)));
            }
            else
            {
                sp->processing_queue.push(
                    std::make_tuple(0.0, std::move(new_resumable), "Start improving solution " + std::to_string(ii.i)));
            }
        },
        learnKernel(ii));

    sgommfi->resume(simulator, simulator.now(), sgommfi);
}
std::unique_ptr<SimIGOMFIData> SimParallelAsynchronousKernelGOMEA::learnKernel(const Individual &ii)
{
    Population &population = *this->population;
    rng = population.getGlobalData<Rng>().get();

    // Get objective ranges
    auto objective_ranges = compute_objective_ranges(population, objective_indices, individuals);

    // Create copies.
    std::vector<Individual> copies(population_size);
    // Index of current solution in population.
    std::optional<size_t> idx_maybe;
    for (size_t i = 0; i < population_size; ++i)
    {
        copies[i] = population.newIndividual();
        if (individuals[i].i == ii.i)
            idx_maybe = i;
        population.copyIndividual(individuals[i], copies[i]);
    }
    t_assert(idx_maybe.has_value(), "Solution being improved should be in population.");
    size_t idx = *idx_maybe;

    // TODO: Fix single-objective directions.
    if (false && objective_indices.size() > 1)
    {
        // Infer clusters - over copies!
        auto clusters = cluster_mo_gomea(population, copies, objective_indices, objective_ranges, number_of_clusters);
        // Determine extreme clusters
        determine_extreme_clusters(objective_indices, clusters);
        determine_cluster_to_use_mo_gomea(population, clusters, copies, objective_indices, objective_ranges);

        // printClusters(clusters);

        // Assign extreme objectives
        TypedGetter<ClusterIndex> gli = population.getDataContainer<ClusterIndex>();
        TypedGetter<UseSingleObjective> guso = population.getDataContainer<UseSingleObjective>();

        // Take cluster index of copy, and apply it to
        ClusterIndex &cli = gli.getData(copies[idx]);
        UseSingleObjective &uso = guso.getData(ii);
        long mixing_mode = clusters[cli.cluster_index].mixing_mode;
        uso.index = mixing_mode;
    }

    // Determine neighborhoods & corresponding FOS.
    // TODO: Make the neighborhood determination & corresponding measure a parameter to Kernel GOMEA.
    TypedGetter<GenotypeCategorical> gc = population.getDataContainer<GenotypeCategorical>();
    bool symmetric = true;

    size_t k = static_cast<size_t>(std::ceil(std::sqrt(population_size)));
    std::vector<size_t> indices(copies.size());
    std::iota(indices.begin(), indices.end(), 0);
    auto neighborhood_indices = getNeighborhoods(
        [&copies, &gc](size_t a, size_t b) {
            GenotypeCategorical gca = gc.getData(copies[a]);
            GenotypeCategorical gcb = gc.getData(copies[b]);

            return hamming_distance(
                gca.genotype.data(), gcb.genotype.data(), std::min(gca.genotype.size(), gcb.genotype.size()));
        },
        *rng,
        indices,
        k,
        symmetric);

    auto &nbi = neighborhood_indices[idx];
    std::vector<Individual> nbii(nbi.size());
    for (size_t i = 0; i < nbii.size(); ++i)
    {
        nbii[i] = copies[nbi[i]];
    }

    foslearner->learnFoS(nbii);

    std::vector<DroppingIndividual> copies_dropping(copies.size());
    for (size_t c = 0; c < copies.size(); ++c)
    {
        copies_dropping[c] = std::move(copies[c]);
    }

    std::unique_ptr<ISamplingDistribution> isd;
    if (donor_search)
    {
        isd = std::make_unique<CategoricalDonorSearchDistribution>(population, nbii);
    }
    else
    {
        isd = std::make_unique<CategoricalPopulationSamplingDistribution>(population, nbii);
    }

    return std::make_unique<SimAsyncKernelData>(std::move(copies_dropping),
                                             std::make_unique<FoS>(foslearner->getFoS()),
                                             std::move(isd),
                                             performance_criterion.get(),
                                             this,
                                             ii);
}
Individual &SimParallelAsynchronousKernelGOMEA::getReplacementSolution(const Individual & /* ii */)
{
    // Get random from archive
    auto &archived = archive->get_archived();
    std::uniform_int_distribution<size_t> idx_d(0, archived.size() - 1);
    return std::ref(archived[idx_d(rng->rng)]);
}
void SimParallelAsynchronousKernelGOMEA::initialize()
{
    Population &population = *this->population;
    individuals.resize(0);

    for (size_t i = 0; i < population_size; ++i)
        individuals.push_back(population.newIndividual());

    initializer->initialize(individuals);

    GObjectiveFunction &objective_function = *population.getGlobalData<GObjectiveFunction>();

    for (size_t idx = 0; idx < individuals.size(); ++idx)
    {
        auto ii = individuals[idx];
        auto ii_ae = population.newIndividual();
        auto &t_ii = tgts->getData(ii);
        t_ii.t = 0;
        population.copyIndividual(ii, ii_ae);
        objective_function.of->evaluate(ii_ae);
        auto &t_ii_ae = tgts->getData(ii_ae);

        // Note: how do we deal with unevaluated solutions?
        // Operations like determining the range & other aspects REALLY don't like this.
        auto process_init_evaluation =
            [this, ii, ii_ae, idx](ISimulator &, double, std::unique_ptr<IResumableSimulated> &) {
                // Population &population = *this->population;
                // population.copyIndividual(ii_ae, ii);
                replace_population_individual(idx, ii_ae, ii);

                archive->try_add(ii);
                --sp->num_workers_busy;

                sp->processing_queue.push(
                    std::make_tuple(0.0,
                                    std::make_unique<FunctionalSimResumable>(
                                        [this, ii, idx](ISimulator &, double, std::unique_ptr<IResumableSimulated> &) {
                                            improveSolution(idx, ii, *sp->simulator);
                                        }),
                                    "Start improving solution " + std::to_string(ii.i)));
            };

        sp->processing_queue.push(
            std::make_tuple(t_ii_ae.t,
                            std::make_unique<FunctionalSimResumable>(std::move(process_init_evaluation)),
                            "Initial evaluation for solution " + std::to_string(ii.i)));
    }
}
size_t SimParallelAsynchronousKernelGOMEA::getNISThreshold(const Individual & /* ii */)
{
    return 1 + static_cast<size_t>(std::floor(std::log2(static_cast<double>(population_size))));
}

APtr<ISamplingDistribution> SimParallelAsynchronousKernelGOMEA::getDistributionForFI(Individual & /* ii */)
{
    Population &population = *this->population;

    auto &archived = archive->get_archived();
    std::uniform_int_distribution<size_t> idx_d(0, archived.size() - 1);
    std::vector<Individual> random_from_archive = {archived[idx_d(rng->rng)]};
    return std::make_unique<CategoricalPopulationSamplingDistribution>(population, random_from_archive);
}
void SimParallelAsynchronousKernelGOMEA::run()
{
    while (true)
    {
        step();
    }
}
void SimParallelAsynchronousKernelGOMEA::step()
{
    if (!initialized)
    {
        initialize();
        initialized = true;
    }
    else
    {
        GenotypeCategoricalData &gcd = *population->getGlobalData<GenotypeCategoricalData>();
        size_t ell = gcd.l;
        for (size_t c = 0; c < population_size * ell; ++c)
        {
            step_usual();
        }
    }
}
void SimParallelAsynchronousKernelGOMEA::replace_population_individual(size_t idx,
                                                                       Individual replacement,
                                                                       Individual in_population)
{
    if (sl != NULL)
    {
        sl->end_span(idx, in_population, 0);
        sl->start_span(idx, replacement, 0);
    }
    population->copyIndividual(replacement, in_population);
}

void SimParallelAsynchronousKernelGOMEA::step_usual()
{
    // Do we wait until, or simply reject if the condition does not hold?
    // i.e. use:
    // sp->simulator.simulate_until([this](){ return sp->num_workers_busy < sp->num_workers; });
    // instead to take larger steps?
    // for now: take small steps.
    while (sp->processing_queue.size() > 0 && sp->num_workers_busy < sp->num_workers)
    {
        ++sp->num_workers_busy;
        auto &[t_cost, resumable, msg] = sp->processing_queue.front();
        // Note: we do not have a descriptive message to use here...
        // Maybe provide one in the processing queue?
        sp->simulator->insert_event(std::move(resumable), sp->simulator->now() + t_cost, msg);
        sp->processing_queue.pop();
    }
    double t_before_step = sp->simulator->now();
    sp->simulator->step();
    double t_after_step = sp->simulator->now();
    if (t_after_step > t_before_step)
    {
        // Time has passed, add content of processing_queue_next_t to the processing queue.
        while (!sp->processing_queue_next_t.empty())
        {
            sp->processing_queue.push(std::move(sp->processing_queue_next_t.front()));
            sp->processing_queue_next_t.pop();
        }
    }
}

// SimParallelAsynchronousGOMEA
SimParallelAsynchronousGOMEA::SimParallelAsynchronousGOMEA(
    std::shared_ptr<SimulatorParameters> sp,
    size_t population_size,
    size_t number_of_clusters,
    std::vector<size_t> objective_indices,
    std::shared_ptr<ISolutionInitializer> initializer,
    std::shared_ptr<FoSLearner> foslearner,
    std::shared_ptr<IPerformanceCriterion> performance_criterion,
    std::shared_ptr<IArchive> archive,
    // std::shared_ptr<GOMEAPlugin> plugin = std::make_shared<GOMEAPlugin>(),
    std::shared_ptr<SpanLogger> sl,
    bool donor_search,
    bool autowrap) :
    SimParallelAsynchronousKernelGOMEA(sp, population_size, number_of_clusters, objective_indices, initializer, foslearner, performance_criterion, archive, sl, donor_search, autowrap)
{
}
std::unique_ptr<SimIGOMFIData> SimParallelAsynchronousGOMEA::learnKernel(const Individual &ii)
{
    Population &population = *this->population;
    rng = population.getGlobalData<Rng>().get();

    // Get objective ranges
    auto objective_ranges = compute_objective_ranges(population, objective_indices, individuals);

    // Create copies.
    std::vector<Individual> copies(population_size);
    // Index of current solution in population.
    std::optional<size_t> idx_maybe;
    for (size_t i = 0; i < population_size; ++i)
    {
        copies[i] = population.newIndividual();
        if (individuals[i].i == ii.i)
            idx_maybe = i;
        population.copyIndividual(individuals[i], copies[i]);
    }
    t_assert(idx_maybe.has_value(), "Solution being improved should be in population.");
    size_t idx = *idx_maybe;

    // TODO: Fix single-objective directions.
    if (false && objective_indices.size() > 1)
    {
        // Infer clusters - over copies!
        auto clusters = cluster_mo_gomea(population, copies, objective_indices, objective_ranges, number_of_clusters);
        // Determine extreme clusters
        determine_extreme_clusters(objective_indices, clusters);
        determine_cluster_to_use_mo_gomea(population, clusters, copies, objective_indices, objective_ranges);

        // printClusters(clusters);

        // Assign extreme objectives
        TypedGetter<ClusterIndex> gli = population.getDataContainer<ClusterIndex>();
        TypedGetter<UseSingleObjective> guso = population.getDataContainer<UseSingleObjective>();

        // Take cluster index of copy, and apply it to
        ClusterIndex &cli = gli.getData(copies[idx]);
        UseSingleObjective &uso = guso.getData(ii);
        long mixing_mode = clusters[cli.cluster_index].mixing_mode;
        uso.index = mixing_mode;
    }

    // Determine FOS.
    if (num_kernel_usages_left <= 0)
    {
        foslearner->learnFoS(copies);
        num_kernel_usages_left = static_cast<long long>(population_size);
    }
    num_kernel_usages_left--;

    std::vector<DroppingIndividual> copies_dropping(copies.size());
    for (size_t c = 0; c < copies.size(); ++c)
    {
        copies_dropping[c] = std::move(copies[c]);
    }

    std::unique_ptr<ISamplingDistribution> isd;
    if (donor_search)
    {
        isd = std::make_unique<CategoricalDonorSearchDistribution>(population, copies);
    }
    else
    {
        isd = std::make_unique<CategoricalPopulationSamplingDistribution>(population, copies);
    }

    return std::make_unique<SimAsyncKernelData>(std::move(copies_dropping),
                                             std::make_unique<FoS>(foslearner->getFoS()),
                                             std::move(isd),
                                             performance_criterion.get(),
                                             this,
                                             ii);
}

// SimAsyncKernelData
SimAsyncKernelData::SimAsyncKernelData(std::vector<DroppingIndividual> &&copies,
                                 APtr<FoS> fos,
                                 APtr<ISamplingDistribution> isd,
                                 APtr<IPerformanceCriterion> ipc,
                                 SimParallelAsynchronousKernelGOMEA *context,
                                 Individual ii) :
    copies(std::move(copies)), fos(std::move(fos)), isd(std::move(isd)), ipc(std::move(ipc)), context(context), ii(ii)
{
}
APtr<FoS> SimAsyncKernelData::getFOSForGOM()
{
    return as_ptr(fos);
}
APtr<ISamplingDistribution> SimAsyncKernelData::getDistributionForGOM()
{
    return as_ptr(isd);
}
APtr<IPerformanceCriterion> SimAsyncKernelData::getPerformanceCriterionForGOM()
{
    return as_ptr(ipc);
}
APtr<FoS> SimAsyncKernelData::getFOSForFI()
{
    return as_ptr(fos);
}
APtr<ISamplingDistribution> SimAsyncKernelData::getDistributionForFI()
{
    return context->getDistributionForFI(ii);
}
APtr<IPerformanceCriterion> SimAsyncKernelData::getPerformanceCriterionForFI()
{
    return as_ptr(ipc);
}
