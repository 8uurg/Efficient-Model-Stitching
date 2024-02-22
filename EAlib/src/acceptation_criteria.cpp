#include "acceptation_criteria.hpp"
#include "cppassert.h"
#include "gomea.hpp"
#include <algorithm>
#include <limits>

SingleObjectiveAcceptanceCriterion::SingleObjectiveAcceptanceCriterion(size_t index) : index(index)
{
}

void SingleObjectiveAcceptanceCriterion::afterRegisterData()
{
    auto &pop = *population;
    t_assert(pop.isRegistered<Objective>(),
             "Objective Acceptance Criterion requires the objective value to be defined.")
}
void SingleObjectiveAcceptanceCriterion::updateCache()
{
    if (cache.has_value())
        return;

    Population &pop = *population;
    cache.emplace(Cache{pop.getDataContainer<Objective>()});
}
short SingleObjectiveAcceptanceCriterion::compare(Individual &a, Individual &b)
{
    updateCache();
    auto &oa = cache->tgo.getData(a);
    auto &ob = cache->tgo.getData(b);

    short result = 0;
    // If a is evaluated AND objective of a is better than b, or the solution b has not been evaluated
    if (index < oa.objectives.size() && (index >= ob.objectives.size() || oa.objectives[index] <= ob.objectives[index]))
        result |= 1;
    // If b is evaluated AND objective of b is better than a, or the solution a has not been evaluated
    if (index < ob.objectives.size() && (index >= oa.objectives.size() || ob.objectives[index] <= oa.objectives[index]))
        result |= 2;
    return result;
}

DominationObjectiveAcceptanceCriterion::DominationObjectiveAcceptanceCriterion(const std::vector<size_t> indices) :
    indices(indices)
{
}

void DominationObjectiveAcceptanceCriterion::afterRegisterData()
{
    auto &pop = *population;
    t_assert(pop.isRegistered<Objective>(),
             "Domination-based Objective Acceptance Criterion requires the objective value to be defined.")
}

short DominationObjectiveAcceptanceCriterion::compare(Individual &a, Individual &b)
{
    auto &pop = *population;
    auto &oa = pop.getData<Objective>(a);
    auto &ob = pop.getData<Objective>(b);

    short result = 3;
    for (size_t index : indices)
    {
        // Deal with uninitialized solutions
        bool is_a_def = oa.objectives.size() > index;
        bool is_b_def = ob.objectives.size() > index;
        if (! is_a_def && ! is_b_def) return 0;
        if (! is_a_def) return 2;
        if (! is_b_def) return 1;

        // Again note: lower is better!
        // if a has an objective that is strictly lower, unset 1
        if (oa.objectives[index] > ob.objectives[index])
            result &= 2;
        // if b has an objective that is strictly lower, unset 2
        if (ob.objectives[index] > oa.objectives[index])
            result &= 1;
    }
    // Effective result is that if equal, neither bit is unset.
    // if one has an objective strictly better, while others, the right bit remains set, while the other is unset.
    return result;
}

SequentialCombineAcceptanceCriterion::SequentialCombineAcceptanceCriterion(
    std::vector<std::shared_ptr<IPerformanceCriterion>> criteria, bool nondeterminable_is_equal) :
    criteria(criteria), nondeterminable_is_equal(nondeterminable_is_equal)
{
}
void SequentialCombineAcceptanceCriterion::setPopulation(std::shared_ptr<Population> population)
{
    for (auto &criterion : criteria)
    {
        criterion->setPopulation(population);
    }
}
void SequentialCombineAcceptanceCriterion::registerData()
{
    for (auto &criterion : criteria)
    {
        criterion->registerData();
    }
}
void SequentialCombineAcceptanceCriterion::afterRegisterData()
{
    for (auto &criterion : criteria)
    {
        criterion->afterRegisterData();
    }
}
short SequentialCombineAcceptanceCriterion::compare(Individual &a, Individual &b)
{
    short r = 0;
    for (auto &criterion : criteria)
    {
        r = criterion->compare(a, b);
        if (r == 3 || (nondeterminable_is_equal && r == 0))
            continue;
        return r;
    }
    return r;
}
ThresholdAcceptanceCriterion::ThresholdAcceptanceCriterion(size_t index, double threshold, bool slack) :
    index(index), threshold(threshold), slack(slack)
{
}
short ThresholdAcceptanceCriterion::compare(Individual &a, Individual &b)
{
    auto &pop = *population;
    auto &oa = pop.getData<Objective>(a);
    auto &ob = pop.getData<Objective>(b);

    // Deal with unevaluated solutions.
    bool is_a_def = oa.objectives.size() > index;
    bool is_b_def = ob.objectives.size() > index;
    if (! is_a_def && ! is_b_def) return 0;
    if (! is_a_def ) return 2;
    if (! is_b_def) return 1;

    if (slack)
    {
        double oa_th = std::max(oa.objectives[index] - threshold, 0.0);
        double ob_th = std::max(ob.objectives[index] - threshold, 0.0);

        short result = 0;
        if (oa_th <= ob_th)
            result |= 1;
        if (ob_th <= oa_th)
            result |= 2;
        return result;
    }
    else
    {
        bool oa_th = oa.objectives[index] > threshold;
        bool ob_th = ob.objectives[index] > threshold;

        short result = 0;
        if (oa_th <= ob_th)
            result |= 1;
        if (ob_th <= oa_th)
            result |= 2;
        return result;
    }
}

double Scalarizer::scalarize(Individual ii)
{
    return this->scalarize_with_weights_of(ii, ii);
}
void Scalarizer::registerData()
{
    if (!scalarizer_id.has_value())
    {
        Population &pop = *population;
        if (!pop.isGlobalRegistered<ScalarizerIndex>())
        {
            pop.registerGlobalData(ScalarizerIndex{});
        }
        auto &si = *pop.getGlobalData<ScalarizerIndex>();
        // Set id
        scalarizer_id.emplace(si.scalarizers.size());
        si.scalarizers.push_back(this);
    }
}

TschebysheffObjectiveScalarizer::TschebysheffObjectiveScalarizer(std::vector<size_t> objective_indices) :
    objective_indices(objective_indices)
{
}
void TschebysheffObjectiveScalarizer::updateCache()
{
    Population &population = *this->population;
    if (cache.has_value()) return;
    
    cache.emplace(
        Cache{population.getDataContainer<ScalarizationWeights>(), population.getDataContainer<Objective>()});
    
}
void TschebysheffObjectiveScalarizer::update_frame_of_reference(std::vector<Individual> individuals)
{
    Population &population = *this->population;
    std::tie(min_v, max_v, ranges) = compute_objective_min_max_ranges(population, objective_indices, individuals);
}
double TschebysheffObjectiveScalarizer::scalarize_with_weights_of(Individual ii, Individual w)
{
    updateCache();
    auto &c = cache.value();
    auto &obj = c.tgo.getData(ii);
    auto &sw = c.tgsw.getData(w);
    size_t scalarization_id = *scalarizer_id;
    t_assert(sw.weights.size() > scalarization_id, "weights should be initialized prior to scalarization - 0");
    auto &weights = sw.weights[*scalarizer_id];
    t_assert(weights.size() == objective_indices.size(), "weights should be initialized prior to scalarization - 1");
    double scalarization = 0.0;
    // std::cout << "Got objectives ";
    for (size_t io = 0; io < objective_indices.size(); ++io)
    {
        size_t o = objective_indices[io];
        // If the solution is not evaluated - return infinity.
        if (obj.objectives.size() <= io) {
            return std::numeric_limits<double>::infinity();
        }
        // std::cout << o << ", ";
        scalarization = std::max(scalarization, weights[io] * ((obj.objectives[o] - min_v[io]) / ranges[io]));
    }
    // std::cout << " scalarized to " << scalarization << "." << std::endl;
    return scalarization;
}

ScalarizationAcceptanceCriterion::ScalarizationAcceptanceCriterion(
    std::shared_ptr<Scalarizer> scalarizer,
    bool use_weights_of_first
) :
    scalarizer(scalarizer), use_weights_of_first(use_weights_of_first)
{
}
void ScalarizationAcceptanceCriterion::setPopulation(std::shared_ptr<Population> population)
{
    scalarizer->setPopulation(population);
}
void ScalarizationAcceptanceCriterion::registerData()
{
    scalarizer->registerData();
}
void ScalarizationAcceptanceCriterion::afterRegisterData()
{
    scalarizer->afterRegisterData();
}
short ScalarizationAcceptanceCriterion::compare(Individual &a, Individual &b)
{
    double sa = scalarizer->scalarize(a);
    double sb = scalarizer->scalarize_with_weights_of(b, use_weights_of_first ? a : b);

    short result = 0;
    if (sa <= sb)
        result |= 1;
    if (sb <= sa)
        result |= 2;
    return result;
}
std::tuple<std::vector<double>, std::vector<double>, std::vector<double>> compute_objective_min_max_ranges(
    Population &population, std::vector<size_t> objective_indices, std::vector<Individual> &pool)
{
    std::vector<double> ranges(objective_indices.size());
    std::vector<double> min_v(objective_indices.size());
    std::fill(min_v.begin(), min_v.end(), std::numeric_limits<double>::infinity());
    std::vector<double> max_v(objective_indices.size());
    std::fill(max_v.begin(), max_v.end(), -std::numeric_limits<double>::infinity());

    for (auto &i : pool)
    {
        auto &obj = population.getData<Objective>(i);
        for (size_t oi = 0; oi < objective_indices.size(); ++oi)
        {
            if (objective_indices[oi] < obj.objectives.size() && !std::isnan(obj.objectives[objective_indices[oi]]))
            {
                auto obj_v = obj.objectives[objective_indices[oi]];
                min_v[oi] = std::min(min_v[oi], obj_v);
                max_v[oi] = std::max(max_v[oi], obj_v);
                ranges[oi] = max_v[oi] - min_v[oi];
            }
        }
    }

    return std::make_tuple(min_v, max_v, ranges);
}

FunctionAcceptanceCriterion::FunctionAcceptanceCriterion(
    std::function<short(Population &pop, Individual &a, Individual &b)> criterion) :
    criterion(std::move(criterion))
{
}

short FunctionAcceptanceCriterion::compare(Individual &a, Individual &b)
{
    return criterion(*population, a, b);
}
