#include "kernels.hpp"

void MetricKernelValuePerformanceCriterion::doCache() {
    if (cache.has_value())
        return;

    Population &pop = *population;

    cache.emplace(Cache{pop.getDataContainer<MetricKernelValue>()});
}
short MetricKernelValuePerformanceCriterion::compare(Individual &a, Individual &b)
{
    doCache();
    Cache &c = *cache;
    auto &da = c.tgmkv.getData(a);
    auto &db = c.tgmkv.getData(b);

    short result = 0;
    // Note: lower is better
    if (da.v <= db.v)
        result |= 1;
    if (da.v >= db.v)
        result |= 2;

    return result;
}
void CachedMetricKernelSelection::doCache()
{
    if (cache.has_value())
        return;

    Population &pop = *population;

    cache.emplace(Cache{pop.getDataContainer<MetricKernelValue>()});
}
CachedMetricKernelSelection::CachedMetricKernelSelection(
    std::function<double(Population &, Individual &, Individual &)> metric, std::shared_ptr<ISelection> selection) :
    metric(metric), selection(selection)
{
}
void CachedMetricKernelSelection::setPopulation(std::shared_ptr<Population> population)
{
    this->population = population;
    selection->setPopulation(population);
}
void CachedMetricKernelSelection::registerData()
{
    Population &pop = *population;
    pop.registerData<MetricKernelValue>();
    selection->registerData();
}
void CachedMetricKernelSelection::afterRegisterData()
{
    selection->afterRegisterData();
}
std::vector<Individual> CachedMetricKernelSelection::select(Individual kernel,
                                                            std::vector<Individual> &ii_population,
                                                            size_t amount)
{
    doCache();

    Cache &c = *cache;

    // Compute metric values
    for (size_t idx = 0; idx < ii_population.size(); ++idx)
    {
        c.tgmkv.getData(ii_population[idx]).v = metric(*population, kernel, ii_population[idx]);
    }

    // Perform selection (hopefully using a method that uses aforementioned metric)
    return selection->select(ii_population, amount);
}
