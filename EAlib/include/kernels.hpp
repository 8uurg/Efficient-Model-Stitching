#pragma once

#include "base.hpp"
#include "ga.hpp"

/**
 * @brief Selection relying on a reference solution.
 *
 */
class IKernelSelection : public IDataUser
{
  public:
    virtual std::vector<Individual> select(Individual kernel,
                                           std::vector<Individual> &ii_population,
                                           size_t amount) = 0;
};

struct MetricKernelValue
{
    double v;
};

class MetricKernelValuePerformanceCriterion : public IPerformanceCriterion
{
  private:
    struct Cache
    {
        TypedGetter<MetricKernelValue> tgmkv;
    };
    std::optional<Cache> cache;

    void doCache();

  public:
    short compare(Individual &a, Individual &b);
};

class CachedMetricKernelSelection : public IKernelSelection
{
  private:
    std::function<double(Population &, Individual &, Individual &)> metric;
    std::shared_ptr<ISelection> selection;

    struct Cache
    {
        TypedGetter<MetricKernelValue> tgmkv;
    };
    std::optional<Cache> cache;

    void doCache();

  public:
    CachedMetricKernelSelection(std::function<double(Population &, Individual &, Individual &)> metric,
                                std::shared_ptr<ISelection> selection);

    void setPopulation(std::shared_ptr<Population> population) override;
    void registerData() override;
    void afterRegisterData() override;

    std::vector<Individual> select(Individual kernel, std::vector<Individual> &ii_population, size_t amount) override;
};
