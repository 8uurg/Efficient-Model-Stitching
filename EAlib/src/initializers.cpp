//  DAEDALUS – Distributed and Automated Evolutionary Deep Architecture Learning with Unprecedented Scalability
// 
// This research code was developed as part of the research programme Open Technology Programme with project number 18373, which was financed by the Dutch Research Council (NWO), Elekta, and Ortec Logiqcare.
// 
// Project leaders: Peter A.N. Bosman, Tanja Alderliesten
// Researchers: Alex Chebykin, Arthur Guijt, Vangelis Kostoulas
// Main code developer: Arthur Guijt

#include "initializers.hpp"
#include "base.hpp"
#include "cppassert.h"
#include <cmath>
#include <cstddef>
#include <random>

void check_alphabet_matches_p(GenotypeCategoricalData &gcd, probabilities &ps)
{
    t_assert(gcd.l == ps.size(), "Number of variables should match");
    t_assert(gcd.alphabet_size.size() == ps.size(), "Number of variables should match");
    bool any_wrong_count = false;
    for (size_t i = 0; i < gcd.alphabet_size.size(); ++i)
    {
        bool right_number_of_probabilities = static_cast<size_t>(gcd.alphabet_size[i]) == ps[i].size();
        if (! right_number_of_probabilities)
        {
            if (! any_wrong_count) {
                std::cerr << "Wrong probabilities for ";
            }
            any_wrong_count = true;
            std::cerr << i << " ";
        }
    }
    if (any_wrong_count)
    {
        std::cerr << "." << std::endl;
    }
    t_assert(! any_wrong_count, "Number of probabilities should match number of possible values.");
}
void check_alphabet_matches_p(GenotypeCategoricalData &gcd, variable_p &p)
{
    if (! p.has_value()) return;
    check_alphabet_matches_p(gcd, *p);
}

bool cumulative_normalize_probabilities(std::vector<double> &p)
{
    double sum = 0.0;
    for (size_t i = 0; i < p.size(); ++i)
    {
        // Values less than 0 are not allowed.
        if (p[i] < 0.0) return false;
        sum += p[i];
        p[i] = sum;
    }
    // Non-finite values are not allowed either.
    if (! std::isfinite(sum)) return false;
    for (size_t i = 0; i < p.size(); ++i)
    {
        p[i] /= sum;
    }
    // Successfully normalized.
    return true;
}

void cumulative_normalize_all_probabilities(probabilities &p)
{
    bool all_successful = true;
    for (auto &v: p)
    {
        bool success = cumulative_normalize_probabilities(v);
        if (!success) all_successful = false;
    }
    t_assert(all_successful, "Invalid probabilities provided. Ensure all values are finite and >= 0.0.");
}
void cumulative_normalize_all_probabilities(variable_p &p)
{
    if (!p.has_value()) return;
    cumulative_normalize_all_probabilities(*p);
}

void CategoricalUniformInitializer::initialize(std::vector<Individual> &iis)
{
    auto &pop = (*population);
    GenotypeCategoricalData &data = *pop.getGlobalData<GenotypeCategoricalData>();
    Rng &rng = *pop.getGlobalData<Rng>();
    for (auto ii : iis)
    {
        GenotypeCategorical &genotype = pop.getData<GenotypeCategorical>(ii);
        genotype.genotype.resize(data.l);
        for (unsigned long long int i = 0; i < data.l; ++i)
        {
            std::uniform_int_distribution<unsigned long long int> gene(0, data.alphabet_size[i] - 1);
            genotype.genotype[i] = static_cast<char>(gene(rng.rng));
        }
    }
}
void CategoricalUniformInitializer::afterRegisterData()
{
    ISolutionInitializer::afterRegisterData();
    Population &pop = (*population);
    t_assert(pop.isRegistered<GenotypeCategorical>(),
             "This initializer requires a categorical genotype to be present.");
}

CategoricalProbabilisticallyCompleteInitializer::CategoricalProbabilisticallyCompleteInitializer(variable_p p) : p(p)
{
    cumulative_normalize_all_probabilities(this->p);
}
void CategoricalProbabilisticallyCompleteInitializer::initialize_uniformly(
    std::vector<Individual> &iis)
{
    auto &pop = (*population);
    Rng &rng = *pop.getGlobalData<Rng>();
    GenotypeCategoricalData &data = *pop.getGlobalData<GenotypeCategoricalData>();
    std::vector<char> genes(iis.size());
    for (unsigned long long int i = 0; i < data.l; ++i)
    {
        // Fill vector of genes such that each count is expressed approximately equally
        for (unsigned long long int j = 0; j < iis.size(); j++)
        {
            genes[j] = static_cast<char>(j % static_cast<unsigned long long int>(data.alphabet_size[i]));
        }
        // And shuffle it!
        std::shuffle(genes.begin(), genes.end(), rng.rng);
        // Assign the genes to each member
        for (unsigned long long int j = 0; j < iis.size(); j++)
        {
            GenotypeCategorical &genotype = pop.getData<GenotypeCategorical>(iis[j]);
            genotype.genotype[i] = genes[j];
        }
    }
}
void CategoricalProbabilisticallyCompleteInitializer::initialize_weighted(
    std::vector<Individual> &iis)
{
    auto &pop = (*population);
    Rng &rng = *pop.getGlobalData<Rng>();
    GenotypeCategoricalData &data = *pop.getGlobalData<GenotypeCategoricalData>();
    std::vector<char> genes(iis.size());
    for (unsigned long long int i = 0; i < data.l; ++i)
    {
        // Fill vector of genes such that each count is expressed accordingly.
        auto &pv = (*p)[i];
        size_t idx = 0;
        for (size_t cv = 0; cv < pv.size(); ++cv)
        {
            double th_double = pv[cv] * static_cast<double>(iis.size());
            size_t th = static_cast<size_t>(std::floor(th_double));
            for (; idx < th; ++idx)
            {
                genes[idx] = static_cast<char>(cv);
            }
        }
        // And shuffle it!
        std::shuffle(genes.begin(), genes.end(), rng.rng);
        // Assign the genes to each member
        for (unsigned long long int j = 0; j < iis.size(); j++)
        {
            GenotypeCategorical &genotype = pop.getData<GenotypeCategorical>(iis[j]);
            genotype.genotype[i] = genes[j];
        }
    }
}

void CategoricalProbabilisticallyCompleteInitializer::initialize(std::vector<Individual> &iis)
{
    auto &pop = (*population);
    GenotypeCategoricalData &data = *pop.getGlobalData<GenotypeCategoricalData>();
    for (auto ii : iis)
    {
        GenotypeCategorical &genotype = pop.getData<GenotypeCategorical>(ii);
        genotype.genotype.resize(data.l);
    }
    if (p.has_value())
    {
        initialize_weighted(iis);
    }
    else
    {
        initialize_uniformly(iis);
    }
}
void CategoricalProbabilisticallyCompleteInitializer::afterRegisterData()
{
    ISolutionInitializer::afterRegisterData();
    Population &pop = (*population);
    t_assert(pop.isRegistered<GenotypeCategorical>(),
             "This initializer requires a categorical genotype to be present.");
    auto gcd = pop.getGlobalData<GenotypeCategoricalData>();
    check_alphabet_matches_p(*gcd, p);
}

CategoricalWeightedInitializer::CategoricalWeightedInitializer(probabilities p) : p(p)
{
    cumulative_normalize_all_probabilities(this->p);
}
char CategoricalWeightedInitializer::sample_gene(double s, size_t idx)
{
    auto &pv = p[idx];

    size_t min_idx = 0;
    size_t max_idx = pv.size() - 1;

    // The list of numbers is ascendiing.
    // We want to find the index of the first number v for which s <= v.
    // If an index is false - we know all indices up to this point are false.
    // If an index is true - we know all indices following are true too.
    while (min_idx != max_idx)
    {
        size_t middle_idx = (max_idx + min_idx) / 2;
        bool is_current = s <= pv[middle_idx];
        if (is_current) {
            max_idx = middle_idx;
        } else {
            min_idx = middle_idx + 1;
        }
    }
    return static_cast<char>(min_idx);
}
void CategoricalWeightedInitializer::initialize(std::vector<Individual> &iis)
{
    auto &pop = (*population);
    GenotypeCategoricalData &data = *pop.getGlobalData<GenotypeCategoricalData>();
    Rng &rng = *pop.getGlobalData<Rng>();
    std::uniform_real_distribution<double> gene_float(0.0, 1.0);
    for (auto ii : iis)
    {
        GenotypeCategorical &genotype = pop.getData<GenotypeCategorical>(ii);
        genotype.genotype.resize(data.l);
        for (unsigned long long int i = 0; i < data.l; ++i)
        {
            double s = gene_float(rng.rng);
            genotype.genotype[i] = sample_gene(s, i);
        }
    }
}
void CategoricalWeightedInitializer::afterRegisterData()
{
    ISolutionInitializer::afterRegisterData();
    Population &pop = (*population);
    t_assert(pop.isRegistered<GenotypeCategorical>(),
             "This initializer requires a categorical genotype to be present.");
    // Genotype Categorical Data
    auto gcd = pop.getGlobalData<GenotypeCategoricalData>();
    check_alphabet_matches_p(*gcd, p);
}
