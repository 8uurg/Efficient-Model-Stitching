//  DAEDALUS – Distributed and Automated Evolutionary Deep Architecture Learning with Unprecedented Scalability
// 
// This research code was developed as part of the research programme Open Technology Programme with project number 18373, which was financed by the Dutch Research Council (NWO), Elekta, and Ortec Logiqcare.
// 
// Project leaders: Peter A.N. Bosman, Tanja Alderliesten
// Researchers: Alex Chebykin, Arthur Guijt, Vangelis Kostoulas
// Main code developer: Arthur Guijt

#pragma once
#include "base.hpp"
#include <optional>

using probabilities = std::vector<std::vector<double>>;
using variable_p = std::optional<probabilities>;

/**
 * Initializes each solution independently and uniformly over the alphabet
 */
class CategoricalUniformInitializer : public ISolutionInitializer
{
  public:
    void initialize(std::vector<Individual> &iis) override;
    void afterRegisterData() override;
};

/**
 * Initializes all solutions such that each gene occurs in (approximately) equal counts
 * or weighted accordingly.
 */
class CategoricalProbabilisticallyCompleteInitializer : public ISolutionInitializer
{
  public:
    variable_p p;
    CategoricalProbabilisticallyCompleteInitializer(variable_p p = std::nullopt);

    void initialize_uniformly(std::vector<Individual> &iis);
    void initialize_weighted(std::vector<Individual> &iis);
    void initialize(std::vector<Individual> &iis) override;
    void afterRegisterData() override;
};

/**
 * Initializes each solution independently weighted over the alphabet
 */
class CategoricalWeightedInitializer : public ISolutionInitializer
{
  public:
    probabilities p;
    CategoricalWeightedInitializer(probabilities p);

    char sample_gene(double s, size_t idx);
    void initialize(std::vector<Individual> &iis) override;
    void afterRegisterData() override;
};