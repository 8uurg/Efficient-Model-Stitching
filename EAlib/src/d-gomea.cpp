//  DAEDALUS – Distributed and Automated Evolutionary Deep Architecture Learning with Unprecedented Scalability
// 
// This research code was developed as part of the research programme Open Technology Programme with project number 18373, which was financed by the Dutch Research Council (NWO), Elekta, and Ortec Logiqcare.
// 
// Project leaders: Peter A.N. Bosman, Tanja Alderliesten
// Researchers: Alex Chebykin, Arthur Guijt, Vangelis Kostoulas
// Main code developer: Arthur Guijt

#include <cmath>
#include <cstddef>
#include <exception>
#include <limits>
#include <memory>
#include <optional>
#include <random>

#include "acceptation_criteria.hpp"
#include "base.hpp"
#include "cppassert.h"
#include "d-gomea.hpp"
#include "scalarize.hpp"
#include "decentralized.hpp"
#include "gomea.hpp"

// (note:evalex) A note on budget limits and asynchronous function evaluations.
// The limiter implementation in this codebase immediately raises an exception if an evaluation
// beyond that of the set limits is being performed. This would induce an immidiate stop in the algorithm.
// Yet, in a parallel setting, there may be evaluations that are still 'in-flight'. Stopping immidiately
// stops us from processing these evaluations. As such we will want to catch these exceptions
// to avoid stopping the entire algorithm.

// -- Reused Generational (Non-Simulated) -> Generational (Simulated)

/**
 * @brief Simulate the parallel initialization of GOMEA.
 *
 * @param sp Base simulator to use.
 * @param base Approach to initialize the population for.
 */
void schedule_init_generation_gomea(std::shared_ptr<Scheduler> &wd, BaseGOMEA *base)
{
    std::unique_ptr<WaitForOtherEventsResumable> wfoer(new WaitForOtherEventsResumable(wd->tag_stack.back()));
    WaitForOtherEventsResumable *wfoer_ = wfoer.get();
    wd->tag_stack.pop_back();
    size_t completion_tag = wd->get_tag(std::move(wfoer));

    Population &population = *base->population;
    base->individuals.resize(base->population_size);

    for (size_t i = 0; i < base->population_size; ++i)
        base->individuals[i] = population.newIndividual();

    base->initializer->initialize(base->individuals);

    GObjectiveFunction &objective_function = *population.getGlobalData<GObjectiveFunction>();

    for (auto ii : base->individuals)
    {
        // Note: Unlike the simulator, some tasks simply have a completion time farther into the future
        //       if the resources are all busy. No need to keep track of active # of processors.
        // Note: Nor do we need to spend resources on managing time.
        // We do however need to keep track of how many are pending, so we know when everything has completed.
        wfoer_->wait_for_one();

        // Note: Evaluation may throw an exception prior to, and after evaluation.
        //  But: not sure how to handle cases where it is prior to where it goes wrong.

        // Register a handler for when function evaluation completes.
        auto tag = wd->get_tag(std::make_unique<FunctionalResumable>([base, ii, completion_tag](Scheduler &wd) {
            // When evaluation completes...

            // Add solution to archive
            base->archive->try_add(ii);

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

/**
 * @brief Schedule a parallel generation of GOMEA.
 *
 * @tparam T A lambda/function pointer of type void(Individual, Simulator &)
 * @param sp Simulator to use
 * @param individuals Population of solutions to perform a generation on
 * @param improveSolution the solution specific improvement operator (i.e. GOM iteration)
 *        Should use tag-completion
 */
template <typename T>
void schedule_step_generation(std::shared_ptr<Scheduler> &wd, std::vector<Individual> &individuals, T &&improveSolution)
{
    static_assert(std::is_invocable<T, size_t, Individual, Scheduler &>::value,
                  "signature improveSolution is not void(Individual, Scheduler &)");

    // Again: make sure that the processing event is scheduled properly.
    std::unique_ptr<WaitForOtherEventsResumable> wfoer(new WaitForOtherEventsResumable(wd->tag_stack.back()));
    WaitForOtherEventsResumable *wfoer_ = wfoer.get();
    wd->tag_stack.pop_back();
    size_t completion_tag = wd->get_tag(std::move(wfoer));

    for (size_t idx = 0; idx < individuals.size(); ++idx)
    {
        auto &ii = individuals[idx];
        // Much like initialization: we do no longer need to keep track of resources in here.
        // We do however need to keep in mind when the actual improvement has finished.
        wfoer_->wait_for_one();
        wd->tag_stack.push_back(completion_tag);

        // Schedule an improvement attempt - note: this should use the tag stack!
        improveSolution(idx, ii, *wd);
    }
}

// -- State Machines --

// StepwiseDistributedGOM
StateMachineGOM::StateMachineGOM(Population &population,
                                 Individual to_improve,
                                 APtr<FoS> fos,
                                 APtr<ISamplingDistribution> distribution,
                                 APtr<IPerformanceCriterion> acceptance_criterion,
                                 std::function<void(Individual &, Individual &)> replace_in_population,
                                 size_t completion_tag,
                                 bool steadystate,
                                 bool *changed,
                                 bool *improved,
                                 bool *limited) :
    population(population),
    to_improve(to_improve),
    backup(population.newIndividual()),
    current(population.newIndividual()),
    fos(std::move(fos)),
    distribution(std::move(distribution)),
    acceptance_criterion(std::move(acceptance_criterion)),
    replace_in_population(replace_in_population),
    completion_tag(completion_tag),
    steadystate(steadystate),
    changed(changed),
    improved(improved),
    limited(limited)
{
    population.copyIndividual(to_improve, *backup);
    population.copyIndividual(to_improve, *current);
    *changed = false;
    *improved = false;
}
std::unique_ptr<StateMachineGOM> StateMachineGOM::apply(
    Population &population,
    Individual to_improve,
    APtr<FoS> fos,
    APtr<ISamplingDistribution> distribution,
    APtr<IPerformanceCriterion> acceptance_criterion,
    std::function<void(Individual &, Individual &)> replace_in_population,
    size_t completion_tag,
    bool steadystate,
    bool *changed,
    bool *improved,
    bool *limited)
{
    return std::make_unique<StateMachineGOM>(population,
                                             to_improve,
                                             std::move(fos),
                                             std::move(distribution),
                                             std::move(acceptance_criterion),
                                             std::move(replace_in_population),
                                             completion_tag,
                                             steadystate,
                                             changed,
                                             improved,
                                             limited);
}

void StateMachineGOM::evaluate_change(Individual current,
                                      Individual /* backup */,
                                      std::vector<size_t> & /* elements_changed */)
{
    //
    GObjectiveFunction &objective_function = *population.getGlobalData<GObjectiveFunction>();
    
    // (note:evalex) - See evaluate above.
    try {
        objective_function.of->evaluate(current);
    } catch (run_complete &e) {
        *limited = true;
    }
    

    // TODO: Add partial evaluations here when supported!
}
bool StateMachineGOM::resume(Scheduler &wd, std::unique_ptr<IResumable> &resumable)
{
    if (*limited) {
        // We are done immidiately: there are no more evaluations to be performed,
        // and we have no evaluation to handle: the last evaluation failed!
        wd.complete_tag(completion_tag);
        return false;
    }
    auto fosp = as_ptr(fos);
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
        // ?: Update of population can also be made more often!
        //  Updating only after a full iteration of GOM is a design decision
        //  that needs to be analysed.
        if (idx == fosp->size() || steadystate)
        {
            // Update actual population
            // population.copyIndividual(*current, to_improve);
            replace_in_population(*current, to_improve);
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

            // Evaluate change
            size_t tag = wd.get_tag(std::move(resumable));
            wd.tag_stack.push_back(tag);

            // (note:evalex) is handled by DistributedGOMThenMaybeFI by catching the exception raised
            evaluate_change(*current, *backup, e);

            // Next step!
            ++idx;
            // simulator.insert_event(std::move(self),
            //                        event_time,
            //                        "GOM on " + std::to_string(to_improve.i) + ": evaluating change for fos element "
            //                        +
            //                            std::to_string(idx));
            break;
        }
    }
    else
    {
        idx++;
    }
    if (idx > fosp->size())
    {
        // We are done.
        wd.complete_tag(completion_tag);
    }
    return false;
}

StateMachineFI::StateMachineFI(Population &population,
                               Individual to_improve,
                               APtr<FoS> fos,
                               APtr<ISamplingDistribution> distribution,
                               APtr<IPerformanceCriterion> acceptance_criterion,
                               std::function<void(Individual &, Individual &)> replace_in_population,
                               size_t completion_tag,
                               bool *changed,
                               bool *improved,
                               bool *limited) :
    population(population),
    to_improve(to_improve),
    backup(population.newIndividual()),
    current(population.newIndividual()),
    fos(std::move(fos)),
    distribution(std::move(distribution)),
    acceptance_criterion(std::move(acceptance_criterion)),
    replace_in_population(replace_in_population),
    completion_tag(completion_tag),
    changed(changed),
    improved(improved),
    limited(limited)
{
    population.copyIndividual(to_improve, *backup);
    population.copyIndividual(to_improve, *current);
    *changed = false;
    *improved = false;
}
std::unique_ptr<StateMachineFI> StateMachineFI::apply(
    Population &population,
    Individual to_improve,
    APtr<FoS> fos,
    APtr<ISamplingDistribution> distribution,
    APtr<IPerformanceCriterion> acceptance_criterion,
    std::function<void(Individual &, Individual &)> replace_in_population,
    size_t completion_tag,
    bool *changed,
    bool *improved,
    bool *limited)
{
    return std::make_unique<StateMachineFI>(population,
                                            to_improve,
                                            std::move(fos),
                                            std::move(distribution),
                                            std::move(acceptance_criterion),
                                            std::move(replace_in_population),
                                            completion_tag,
                                            changed,
                                            improved,
                                            limited);
}

void StateMachineFI::evaluate_change(Individual current,
                                     Individual /* backup */,
                                     std::vector<size_t> & /* elements_changed */)
{
    //
    GObjectiveFunction &objective_function = *population.getGlobalData<GObjectiveFunction>();
    
    try {
        objective_function.of->evaluate(current);
    } catch (run_complete &e) {
        *limited = true;
    }

    // TODO: Add partial evaluations here when supported!
}
bool StateMachineFI::resume(Scheduler &wd, std::unique_ptr<IResumable> &self)
{
    if (*limited) {
        // We are done immidiately: there are no more evaluations to be performed,
        // and we have no evaluation to handle: the last evaluation failed!
        wd.complete_tag(completion_tag);
        return false;
    }
    auto fosp = as_ptr(fos);
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

            // Set handler for completion of evaluation.
            size_t tag = wd.get_tag(std::move(self));
            wd.tag_stack.push_back(tag);

            // Request evaluation of the change
            evaluate_change(*current, *backup, e);

            // Next fos element - maybe
            ++idx;
            break;
        }
    }
    else
    {
        idx++;
    }
    if (idx > fosp->size() || success)
    {
        // We are done.
        wd.complete_tag(completion_tag);
    }
    return false;
}

// DistributedGOMThenMaybeFI
DistributedGOMThenMaybeFI::DistributedGOMThenMaybeFI(
    Population &population,
    Individual to_improve,
    size_t completion_tag,
    std::function<size_t(Individual &ii)> getNISThreshold,
    std::function<Individual(Individual &ii)> getReplacementSolution,
    std::function<void(Individual &, Individual &)> replace_in_population,
    std::function<void(Scheduler &scheduler)> onCompletion,
    std::unique_ptr<IGOMFIData> kernel_data,
    bool steadystate) :
    population(population),
    to_improve(to_improve),
    kernel_data(std::move(kernel_data)),
    getNISThreshold(getNISThreshold),
    getReplacementSolution(getReplacementSolution),
    replace_in_population(replace_in_population),
    onCompletion(onCompletion),
    completion_tag(completion_tag),
    steadystate(steadystate)
{
    improved = std::make_unique<bool>(false);
    changed = std::make_unique<bool>(false);
    limited = std::make_unique<bool>(false);
}

bool DistributedGOMThenMaybeFI::resume(Scheduler &wd, std::unique_ptr<IResumable> &self)
{
    // if we have stopped due to hitting a limit, just return false.
    if (*limited) return false;

    return progress(wd, self);
}

bool DistributedGOMThenMaybeFI::progress(Scheduler &wd, std::unique_ptr<IResumable> &self)
{

    switch (state)
    {
    case Start: {
        // Re-tag self.
        size_t completion_tag = wd.get_tag(std::move(self));
        // Register GOM
        auto gom = StateMachineGOM::apply(population,
                                          to_improve,
                                          this->kernel_data->getFOSForGOM(),
                                          this->kernel_data->getDistributionForGOM(),
                                          this->kernel_data->getPerformanceCriterionForGOM(),
                                          this->replace_in_population,
                                          completion_tag,
                                          steadystate,
                                          changed.get(),
                                          improved.get(),
                                          limited.get());

        // Immediately request GOM to be performed.
        wd.schedule_immediately(std::move(gom), 0.0);

        state = GOM;
    }
    break;
    case GOM: {
        // Update NIS.
        NIS &nis = population.getData<NIS>(to_improve);
        if (!(*improved))
            nis.nis += 1;

        if (*improved)
        {
            onImprovedSolution(to_improve);
        }

        // If solution hasn't changed, or the NIS threshold has been reached
        // perform Forced Improvements
        if (!(*changed) || nis.nis > getNISThreshold(to_improve))
        {

            // Re-tag self.
            size_t completion_tag = wd.get_tag(std::move(self));

            auto fi = StateMachineFI::apply(population,
                                            to_improve,
                                            kernel_data->getFOSForFI(),
                                            kernel_data->getDistributionForFI(),
                                            kernel_data->getPerformanceCriterionForFI(),
                                            this->replace_in_population,
                                            completion_tag,
                                            changed.get(),
                                            improved.get(),
                                            limited.get());
            wd.schedule_immediately(std::move(fi), 0.0);
            state = FI;
        }
        else
        {
            onCompletion(wd);
            wd.complete_tag(completion_tag);

            state = Completed;
        }
    }
    break;
    case FI:
        if (*improved)
        {
            onImprovedSolution(to_improve);
        }
        if (!(*changed))
        {
            population.copyIndividual(getReplacementSolution(to_improve), to_improve);
        }
        onCompletion(wd);
        wd.complete_tag(completion_tag);
        state = Completed;
        break;
    case Completed:
        break;
    }
    return false;
}
void DistributedGOMThenMaybeFI::onImprovedSolution(Individual &ii)
{
    // Reset NIS
    NIS &nis = population.getData<NIS>(ii);
    nis.nis = 0;
}

// DistributedParallelSynchronousGOMEA
DistributedSynchronousGOMEA::DistributedSynchronousGOMEA(
    std::shared_ptr<Scheduler> wd,
    size_t population_size,
    std::shared_ptr<ISolutionInitializer> initializer,
    std::shared_ptr<FoSLearner> foslearner,
    std::shared_ptr<IPerformanceCriterion> performance_criterion,
    std::shared_ptr<IArchive> archive,
    bool autowrap,
    bool steadystate,
    std::function<APtr<ISamplingDistribution>(Population &, GenerationalData &, BaseGOMEA *)>
        gomSamplingDistributionFactory,
    std::function<APtr<ISamplingDistribution>(Population &, GenerationalData &, BaseGOMEA *)>
        fiSamplingDistributionFactory) :
    GOMEA(population_size,
          initializer,
          foslearner,
          performance_criterion,
          archive,
          autowrap,
          gomSamplingDistributionFactory,
          fiSamplingDistributionFactory),
    wd(std::move(wd)),
    steadystate(steadystate)
{
}
void DistributedSynchronousGOMEA::setPopulation(std::shared_ptr<Population> population)
{
    GOMEA::setPopulation(population);
    wd->setPopulation(population);
    // if (sl != NULL)
    //     sl->setPopulation(population);
}
void DistributedSynchronousGOMEA::registerData()
{
    GOMEA::registerData();
    wd->registerData();
    // if (sl != NULL)
    //     sl->registerData();
    // Population &pop = *this->population;
}
void DistributedSynchronousGOMEA::afterRegisterData()
{
    GOMEA::afterRegisterData();
    wd->afterRegisterData();
    // if (sl != NULL)
    //     sl->afterRegisterData();
}

void DistributedSynchronousGOMEA::step_normal_generation()
{
    this->atGenerationStart();

    size_t generation_complete_tag = wd->tag_stack.back();
    wd->tag_stack.pop_back();

    size_t all_solutions_improved_tag =
        wd->get_tag(std::make_unique<FunctionalResumable>([this, generation_complete_tag](Scheduler &wd) {
            this->atGenerationEnd();

            wd.complete_tag(generation_complete_tag);

            return false;
        }));
    wd->tag_stack.push_back(all_solutions_improved_tag);

    schedule_step_generation(
        wd, individuals, [this](size_t idx, Individual ii, Scheduler &wd) { improveSolution(idx, ii, wd); });
}
void DistributedSynchronousGOMEA::initialize()
{
    auto tag = wd->get_tag(std::make_unique<GenerationalStarter>([this](Scheduler & /* wd */) {
        this->step_normal_generation();
        return false;
    }));
    wd->tag_stack.push_back(tag);

    schedule_init_generation_gomea(wd, this);
}
void DistributedSynchronousGOMEA::improveSolution(size_t idx, Individual ii, Scheduler &wd)
{
    size_t completion_tag = wd.tag_stack.back();
    wd.tag_stack.pop_back();

    std::unique_ptr<IResumable> sgommfi = std::make_unique<DistributedGOMThenMaybeFI>(
        *population,
        ii,
        completion_tag,
        [this, ii](Individual &) { return getNISThreshold(ii); },
        [this, ii](Individual &) { return getReplacementSolution(ii); },
        [this, idx](Individual &replacement, Individual &in_population) {
            return replace_population_individual(idx, replacement, in_population);
        },
        [](Scheduler &) {},
        std::make_unique<GOMFIDataBaseGOMEA>(this, ii),
        steadystate);

    wd.schedule_immediately(std::move(sgommfi), 0.0);
}
void DistributedSynchronousGOMEA::replace_population_individual(size_t /*idx*/,
                                                                Individual replacement,
                                                                Individual in_population)
{
    population->copyIndividual(replacement, in_population);
}

// DistributedParallelSynchronousMO_GOMEA
DistributedSynchronousMO_GOMEA::DistributedSynchronousMO_GOMEA(
    std::shared_ptr<Scheduler> wd,
    size_t population_size,
    size_t number_of_clusters,
    std::vector<size_t> objective_indices,
    std::shared_ptr<ISolutionInitializer> initializer,
    std::shared_ptr<FoSLearner> foslearner,
    std::shared_ptr<IPerformanceCriterion> performance_criterion,
    std::shared_ptr<IArchive> archive,
    std::shared_ptr<GOMEAPlugin> plugin,
    bool donor_search,
    bool autowrap,
    bool steadystate) :
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
    wd(std::move(wd)),
    steadystate(steadystate)
{
}
void DistributedSynchronousMO_GOMEA::setPopulation(std::shared_ptr<Population> population)
{
    MO_GOMEA::setPopulation(population);
    wd->setPopulation(population);
    // if (sl != NULL)
    //     sl->setPopulation(population);
}
void DistributedSynchronousMO_GOMEA::registerData()
{
    MO_GOMEA::registerData();
    wd->registerData();
    // if (sl != NULL)
    //     sl->registerData();
}
void DistributedSynchronousMO_GOMEA::afterRegisterData()
{
    MO_GOMEA::afterRegisterData();
    wd->afterRegisterData();
    // if (sl != NULL)
    //     sl->afterRegisterData();
}
void DistributedSynchronousMO_GOMEA::initialize()
{
    auto tag = wd->get_tag(std::make_unique<GenerationalStarter>([this](Scheduler & /* wd */) {
        this->step_normal_generation();
        return false;
    }));
    wd->tag_stack.push_back(tag);

    schedule_init_generation_gomea(wd, this);
}
void DistributedSynchronousMO_GOMEA::step_normal_generation()
{
    this->atGenerationStart();

    size_t generation_complete_tag = wd->tag_stack.back();
    wd->tag_stack.pop_back();

    size_t all_solutions_improved_tag =
        wd->get_tag(std::make_unique<FunctionalResumable>([this, generation_complete_tag](Scheduler &wd) {
            this->atGenerationEnd();

            wd.complete_tag(generation_complete_tag);

            return false;
        }));
    wd->tag_stack.push_back(all_solutions_improved_tag);

    schedule_step_generation(
        wd, individuals, [this](size_t idx, Individual ii, Scheduler &wd) { improveSolution(idx, ii, wd); });
}
void DistributedSynchronousMO_GOMEA::improveSolution(size_t idx, Individual ii, Scheduler &wd)
{
    size_t completion_tag = wd.tag_stack.back();
    wd.tag_stack.pop_back();

    std::unique_ptr<IResumable> sgommfi = std::make_unique<DistributedGOMThenMaybeFI>(
        *population,
        ii,
        completion_tag,
        [this, ii](Individual &) { return getNISThreshold(ii); },
        [this, ii](Individual &) { return getReplacementSolution(ii); },
        [this, idx](Individual &replacement, Individual &in_population) {
            return replace_population_individual(idx, replacement, in_population);
        },
        [](Scheduler &) {},
        std::make_unique<GOMFIDataBaseGOMEA>(this, ii),
        steadystate);

    wd.schedule_immediately(std::move(sgommfi), 0.0);
}
void DistributedSynchronousMO_GOMEA::replace_population_individual(size_t /*idx*/,
                                                                   Individual replacement,
                                                                   Individual in_population)
{
    // if (sl != NULL)
    // {
    //     sl->end_span(idx, in_population, 0);
    //     sl->start_span(idx, replacement, 0);
    // }
    population->copyIndividual(replacement, in_population);
}

// DistributedParallelSynchronousKernelGOMEA
DistributedSynchronousKernelGOMEA::DistributedSynchronousKernelGOMEA(
    std::shared_ptr<Scheduler> wd,
    size_t population_size,
    size_t number_of_clusters,
    std::vector<size_t> objective_indices,
    std::shared_ptr<ISolutionInitializer> initializer,
    std::shared_ptr<FoSLearner> foslearner,
    std::shared_ptr<IPerformanceCriterion> performance_criterion,
    std::shared_ptr<IArchive> archive,
    std::shared_ptr<GOMEAPlugin> plugin,
    bool donor_search,
    bool autowrap,
    bool steadystate) :
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
    wd(std::move(wd)),
    steadystate(steadystate)
{
}
void DistributedSynchronousKernelGOMEA::setPopulation(std::shared_ptr<Population> population)
{
    KernelGOMEA::setPopulation(population);
    wd->setPopulation(population);
}
void DistributedSynchronousKernelGOMEA::registerData()
{
    KernelGOMEA::registerData();
    wd->registerData();
}
void DistributedSynchronousKernelGOMEA::afterRegisterData()
{
    KernelGOMEA::afterRegisterData();
    wd->afterRegisterData();
}
void DistributedSynchronousKernelGOMEA::initialize()
{
    auto tag = wd->get_tag(std::make_unique<GenerationalStarter>([this](Scheduler & /* wd */) {
        this->step_normal_generation();
        return false;
    }));
    wd->tag_stack.push_back(tag);

    schedule_init_generation_gomea(wd, this);
}
void DistributedSynchronousKernelGOMEA::step_normal_generation()
{
    this->atGenerationStart();

    size_t generation_complete_tag = wd->tag_stack.back();
    wd->tag_stack.pop_back();

    size_t all_solutions_improved_tag =
        wd->get_tag(std::make_unique<FunctionalResumable>([this, generation_complete_tag](Scheduler &wd) {
            this->atGenerationEnd();

            wd.complete_tag(generation_complete_tag);

            return false;
        }));
    wd->tag_stack.push_back(all_solutions_improved_tag);

    schedule_step_generation(
        wd, individuals, [this](size_t idx, Individual ii, Scheduler &wd) { improveSolution(idx, ii, wd); });
}
void DistributedSynchronousKernelGOMEA::replace_population_individual(size_t /*idx*/,
                                                                      Individual replacement,
                                                                      Individual in_population)
{
    population->copyIndividual(replacement, in_population);
}
void DistributedSynchronousKernelGOMEA::improveSolution(size_t idx, Individual ii, Scheduler &wd)
{

    size_t completion_tag = wd.tag_stack.back();
    wd.tag_stack.pop_back();

    std::unique_ptr<IResumable> sgommfi = std::make_unique<DistributedGOMThenMaybeFI>(
        *population,
        ii,
        completion_tag,
        [this, ii](Individual &) { return getNISThreshold(ii); },
        [this, ii](Individual &) { return getReplacementSolution(ii); },
        [this, idx](Individual &replacement, Individual &in_population) {
            return replace_population_individual(idx, replacement, in_population);
        },
        [](Scheduler &) {},
        std::make_unique<GOMFIDataBaseGOMEA>(this, ii),
        steadystate);
}

// DistributedAsynchronousBaseGOMEA
DistributedAsynchronousBaseGOMEA::DistributedAsynchronousBaseGOMEA(
    std::shared_ptr<Scheduler> wd,
    size_t population_size,
    size_t number_of_clusters,
    std::vector<size_t> objective_indices,
    std::shared_ptr<ISolutionInitializer> initializer,
    std::shared_ptr<FoSLearner> foslearner,
    std::shared_ptr<IPerformanceCriterion> performance_criterion,
    std::shared_ptr<IArchive> archive,
    // std::shared_ptr<GOMEAPlugin> plugin = std::make_shared<GOMEAPlugin>(),
    bool donor_search,
    bool autowrap,
    bool steadystate,
    bool sample_from_copies) :
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
    wd(std::move(wd)),
    steadystate(steadystate),
    sample_from_copies(sample_from_copies)
{
}
void DistributedAsynchronousBaseGOMEA::setPopulation(std::shared_ptr<Population> population)
{
    IDataUser::setPopulation(population);
    wd->setPopulation(population);
    initializer->setPopulation(population);
    foslearner->setPopulation(population);
    performance_criterion->setPopulation(population);
    archive->setPopulation(population);
}
void DistributedAsynchronousBaseGOMEA::registerData()
{
    wd->registerData();
    initializer->registerData();
    foslearner->registerData();
    performance_criterion->registerData();
    archive->registerData();

    Population &pop = *this->population;
    pop.registerData<ClusterIndex>();
    pop.registerData<UseSingleObjective>();
    pop.registerData<NIS>();

    // If there are scalarizers registered - we will also generate scalarization weights.
    if (pop.isGlobalRegistered<ScalarizerIndex>()) {
        pop.registerData<ScalarizationWeights>();
    }
}
void DistributedAsynchronousBaseGOMEA::afterRegisterData()
{
    wd->afterRegisterData();
    initializer->afterRegisterData();
    foslearner->afterRegisterData();
    performance_criterion->afterRegisterData();
    archive->afterRegisterData();

    Population &population = *this->population;
    rng = population.getGlobalData<Rng>().get();
}
void DistributedAsynchronousBaseGOMEA::onImprovedSolution(Individual &ii)
{
    Population &population = *this->population;
    // Reset NIS
    NIS &nis = population.getData<NIS>(ii);
    nis.nis = 0;
}

void DistributedAsynchronousBaseGOMEA::improveSolution(size_t idx, const Individual &ii, Scheduler &wd)
{
    size_t qmh = wd.num_queue_messages_handled;

    size_t completion_tag = wd.get_tag(nullptr);
    // wd.tag_stack.pop_back();

    std::unique_ptr<IResumable> sgommfi = std::make_unique<DistributedGOMThenMaybeFI>(
        *population,
        ii,
        completion_tag,
        [this, ii](Individual &) { return getNISThreshold(ii); },
        [this, ii](Individual &) { return getReplacementSolution(ii); },
        [this, idx](Individual &replacement, Individual &in_population) {
            return replace_population_individual(idx, replacement, in_population);
        },
        [this, ii, qmh, idx](Scheduler &wd) {
            auto new_resumable = std::make_unique<FunctionalResumable>([ii, this, idx](Scheduler &wd) {
                improveSolution(idx, ii, wd);

                return false;
            });

            size_t qmh_now = wd.num_queue_messages_handled;

            if (qmh == qmh_now)
            {
                // Time at which we were scheduled to be processed == time of completion.
                // Which means no time has passed (and scheduling it to process again will likely not
                // yield any different results: no time has passed after all)
                // Alternatively, if things have changed, this SHOULD have advanced the clock,
                // this is a bit of an assumption however: if things change without needing
                // evaluations, this could introduce unnecessary waiting time.
                // However: this is frankly not the end of the world, especially compared to time
                // getting stuck indefinitely.
                // std::cerr << "Warning: GOM & FI were applied but returned without performing any evaluations." << std::endl;
                // std::cerr << "We have likely converged." << std::endl;
                wd.schedule_after_message_completion(std::move(new_resumable), 0.0);
            }
            else
            {
                // Otherwise, we can schedule immediately.
                wd.schedule_immediately(std::move(new_resumable), 0.0);
            }
        },
        learnKernel(ii),
        steadystate);

    wd.schedule_immediately(std::move(sgommfi), 0.0);
}

DistributedAsynchronousKernelGOMEA::DistributedAsynchronousKernelGOMEA(
    std::shared_ptr<Scheduler> wd,
    size_t population_size,
    size_t number_of_clusters,
    std::vector<size_t> objective_indices,
    std::shared_ptr<ISolutionInitializer> initializer,
    std::shared_ptr<FoSLearner> foslearner,
    std::shared_ptr<IPerformanceCriterion> performance_criterion,
    std::shared_ptr<IArchive> archive,
    // std::shared_ptr<GOMEAPlugin> plugin = std::make_shared<GOMEAPlugin>(),
    bool donor_search,
    bool autowrap,
    bool steadystate,
    bool sample_from_copies,
    std::optional<std::shared_ptr<LKStrategy>> lkg_strategy) :
    DistributedAsynchronousBaseGOMEA(wd,
                                     population_size,
                                     number_of_clusters,
                                     objective_indices,
                                     initializer,
                                     foslearner,
                                     performance_criterion,
                                     archive,
                                     donor_search,
                                     autowrap,
                                     steadystate,
                                     sample_from_copies),
    lkg_strategy(lkg_strategy.value_or(std::make_shared<SimpleStrategy>(LKSimpleStrategy::SQRT_SYM)))
{
}

std::unique_ptr<IGOMFIData> DistributedAsynchronousKernelGOMEA::learnKernel(const Individual &ii)
{
    Population &population = *this->population;
    rng = population.getGlobalData<Rng>().get();

    // Get objective ranges
    auto objective_ranges = compute_objective_ranges(population, objective_indices, individuals);
    ensure_valid_ranges(objective_ranges);

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

    // TODO Do this less frequently?
    assign_scalarization_weights_if_necessary(population, individuals);
    // TODO: Fix single-objective directions - currently they may be overriden.
    bool is_multiobjective = objective_indices.size() > 1;
    if (is_multiobjective)
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
    auto subset = std::vector<size_t> { idx };
    auto neighborhood_indices = lkg_strategy->determine_lk_neighborhood(
        population,
        copies,
        subset,
        population_size
    );

    auto &nbi = neighborhood_indices[0];
    std::vector<Individual> nbii(nbi.size());
    for (size_t i = 0; i < nbii.size(); ++i)
    {
        nbii[i] = individuals[nbi[i]];
    }

    foslearner->learnFoS(nbii);

    std::unique_ptr<ISamplingDistribution> isd;
    if (donor_search)
    {
        if (sample_from_copies)
        {
            isd = std::make_unique<CategoricalStoredDonorSearchDistribution>(population, nbii);
        }
        else
        {
            isd = std::make_unique<CategoricalDonorSearchDistribution>(population, nbii);
        }
    }
    else
    {
        if (sample_from_copies)
        {
            isd = std::make_unique<CategoricalStoredSamplingDistribution>(population, nbii);
        }
        else
        {
            isd = std::make_unique<CategoricalPopulationSamplingDistribution>(population, nbii);
        }
    }
    
    // Much like the original implementation, clean up the copies.
    std::vector<DroppingIndividual> copies_dropping(copies.size());
    for (size_t c = 0; c < copies.size(); ++c)
    {
        copies_dropping[c] = std::move(copies[c]);
    }
    copies_dropping.resize(0);

    return std::make_unique<AsyncKernelData>(std::move(copies_dropping),
                                             std::make_unique<FoS>(foslearner->getFoS()),
                                             std::move(isd),
                                             performance_criterion.get(),
                                             this,
                                             ii);
}
Individual &DistributedAsynchronousBaseGOMEA::getReplacementSolution(const Individual & /* ii */)
{
    // Get random from archive
    auto &archived = archive->get_archived();
    std::uniform_int_distribution<size_t> idx_d(0, archived.size() - 1);
    return std::ref(archived[idx_d(rng->rng)]);
}
void DistributedAsynchronousBaseGOMEA::initialize()
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

        population.copyIndividual(ii, ii_ae);

        // Note: how do we deal with unevaluated solutions?
        // Operations like determining the range & other aspects REALLY don't like this.
        auto process_init_evaluation = [this, ii, ii_ae, idx](Scheduler &wd) {
            if (wd.get_exception() != nullptr) {
                // Evaluation has failed due to budget issues. :)
                return false;
            }
            // Population &population = *this->population;
            // population.copyIndividual(ii_ae, ii);
            replace_population_individual(idx, ii_ae, ii);

            archive->try_add(ii);

            // TODO: Notify for completion?

            wd.schedule_immediately(std::make_unique<FunctionalResumable>([this, ii, idx](Scheduler &wd) {
                                        improveSolution(idx, ii, wd);
                                        return false;
                                    }),
                                    0.0);

            return false;
        };

        size_t evaluation_complete_tag =
            wd->get_tag(std::make_unique<FunctionalResumable>(std::move(process_init_evaluation)));
        wd->tag_stack.push_back(evaluation_complete_tag);

        try {
            objective_function.of->evaluate(ii_ae);
        } catch (run_complete &e) {
            // See (note:evalex)
            auto exptr = std::current_exception();
            wd->complete_tag(evaluation_complete_tag, exptr);
        }
    }
}
size_t DistributedAsynchronousBaseGOMEA::getNISThreshold(const Individual & /* ii */)
{
    return 1 + static_cast<size_t>(std::floor(std::log2(static_cast<double>(population_size))));
}

APtr<ISamplingDistribution> DistributedAsynchronousBaseGOMEA::getDistributionForFI(Individual & /* ii */)
{
    Population &population = *this->population;

    auto &archived = archive->get_archived();
    std::uniform_int_distribution<size_t> idx_d(0, archived.size() - 1);
    std::vector<Individual> random_from_archive = {archived[idx_d(rng->rng)]};
    return std::make_unique<CategoricalPopulationSamplingDistribution>(population, random_from_archive);
}
void DistributedAsynchronousBaseGOMEA::run()
{
    while (true)
    {
        step();
    }
}
// void DistributedAsynchronousBaseGOMEA::step()
// {
//     if (!initialized)
//     {
//         initialize();
//         initialized = true;
//     }
//     else
//     {
//         GenotypeCategoricalData &gcd = *population->getGlobalData<GenotypeCategoricalData>();
//         size_t ell = gcd.l;
//         for (size_t c = 0; c < population_size * ell; ++c)
//         {
//             step_usual();
//         }
//     }
// }
void DistributedAsynchronousBaseGOMEA::replace_population_individual(size_t /* idx */,
                                                                       Individual replacement,
                                                                       Individual in_population)
{
    population->copyIndividual(replacement, in_population);
}

// DistributedAsynchronousGOMEA
DistributedAsynchronousGOMEA::DistributedAsynchronousGOMEA(
    std::shared_ptr<Scheduler> wd,
    size_t population_size,
    size_t number_of_clusters,
    std::vector<size_t> objective_indices,
    std::shared_ptr<ISolutionInitializer> initializer,
    std::shared_ptr<FoSLearner> foslearner,
    std::shared_ptr<IPerformanceCriterion> performance_criterion,
    std::shared_ptr<IArchive> archive,
    // std::shared_ptr<GOMEAPlugin> plugin = std::make_shared<GOMEAPlugin>(),
    bool donor_search,
    bool autowrap,
    bool steadystate,
    bool sample_from_copies) :
    DistributedAsynchronousBaseGOMEA(wd,
                                       population_size,
                                       number_of_clusters,
                                       objective_indices,
                                       initializer,
                                       foslearner,
                                       performance_criterion,
                                       archive,
                                       donor_search,
                                       autowrap,
                                       steadystate,
                                       sample_from_copies)
{
}

void DistributedAsynchronousGOMEA::registerData()
{
    DistributedAsynchronousBaseGOMEA::registerData();
}

std::unique_ptr<IGOMFIData> DistributedAsynchronousGOMEA::learnKernel(const Individual &ii)
{
    Population &population = *this->population;
    rng = population.getGlobalData<Rng>().get();

    // Get objective ranges
    auto objective_ranges = compute_objective_ranges(population, objective_indices, individuals);
    ensure_valid_ranges(objective_ranges);

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
    
    if (num_kernel_usages_left <= 0)
    {
        assign_scalarization_weights_if_necessary(population, individuals);
    }
    // Note: this method may behave unexpectedly if not sufficiently many solutions have set their fitness.
    // However, this affects at most a few solutions - as such it is probably fine!
    bool is_multiobjective = objective_indices.size() > 1;
    if (is_multiobjective)
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

    std::unique_ptr<ISamplingDistribution> isd;
    if (donor_search)
    {
        if (sample_from_copies)
        {
            isd = std::make_unique<CategoricalStoredDonorSearchDistribution>(population, copies);
        }
        else
        {
            isd = std::make_unique<CategoricalDonorSearchDistribution>(population, individuals);
        }
    }
    else
    {
        if (sample_from_copies)
        {
            isd = std::make_unique<CategoricalStoredSamplingDistribution>(population, copies);
        }
        else
        {
            isd = std::make_unique<CategoricalPopulationSamplingDistribution>(population, individuals);
        }
    }

    // Previously, we would do this.
    std::vector<DroppingIndividual> copies_dropping(copies.size());
    for (size_t c = 0; c < copies.size(); ++c)
    {
        copies_dropping[c] = std::move(copies[c]);
    }

    // To save memory - we have used a sampling distribution that caches the genotypes only,
    // and does not store a reference to the full solutions incl. metadata.
    // As such, the copies have become unused (hopefully!).
    copies_dropping.resize(0);
    
    return std::make_unique<AsyncKernelData>(std::move(copies_dropping),
                                             std::make_unique<FoS>(foslearner->getFoS()),
                                             std::move(isd),
                                             performance_criterion.get(),
                                             this,
                                             ii);
}

// AsyncKernelData
AsyncKernelData::AsyncKernelData(std::vector<DroppingIndividual> &&copies,
                                 APtr<FoS> fos,
                                 APtr<ISamplingDistribution> isd,
                                 APtr<IPerformanceCriterion> ipc,
                                 DistributedAsynchronousBaseGOMEA *context,
                                 Individual ii) :
    copies(std::move(copies)), fos(std::move(fos)), isd(std::move(isd)), ipc(std::move(ipc)), context(context), ii(ii)
{
}
APtr<FoS> AsyncKernelData::getFOSForGOM()
{
    return as_ptr(fos);
}
APtr<ISamplingDistribution> AsyncKernelData::getDistributionForGOM()
{
    return as_ptr(isd);
}
APtr<IPerformanceCriterion> AsyncKernelData::getPerformanceCriterionForGOM()
{
    return as_ptr(ipc);
}
APtr<FoS> AsyncKernelData::getFOSForFI()
{
    return as_ptr(fos);
}
APtr<ISamplingDistribution> AsyncKernelData::getDistributionForFI()
{
    return context->getDistributionForFI(ii);
}
APtr<IPerformanceCriterion> AsyncKernelData::getPerformanceCriterionForFI()
{
    return as_ptr(ipc);
}
