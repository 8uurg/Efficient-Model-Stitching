//  DAEDALUS – Distributed and Automated Evolutionary Deep Architecture Learning with Unprecedented Scalability
// 
// This research code was developed as part of the research programme Open Technology Programme with project number 18373, which was financed by the Dutch Research Council (NWO), Elekta, and Ortec Logiqcare.
// 
// Project leaders: Peter A.N. Bosman, Tanja Alderliesten
// Researchers: Alex Chebykin, Arthur Guijt, Vangelis Kostoulas
// Main code developer: Arthur Guijt

#include "acceptation_criteria.hpp"
#include "utilities.hpp"

/**
 * @brief Check if there are any uninitialized scalarization weights for this set of individuals.
 * 
 * @param sw TypedGetter for obtaining the scalarization
 * @param individuals the individuals to check
 * @param scalarizer_id the id of the scalarizer to check for
 * @param dim how many elements a full weight vector for this scalarizer should have
 * @return int 
 */
int is_any_individual_scalarizationweight_uninit(
    TypedGetter<ScalarizationWeights> &sw,
    std::vector<Individual> &individuals,
    size_t scalarizer_id,
    size_t dim
);


// For future notes on these methods, please refer to the documentation of
// HoangScalarizationScheme.
void hoang_assign_weights(
    Population &population,
    Scalarizer &scalarizer,
    std::vector<Individual> &individuals,
    Matrix<double> &vectors,
    std::vector<size_t> &vector_indices,
    size_t dim);

void hoang_generate_initial_weights(
    Population &population,
    Scalarizer &scalarizer,
    size_t dim,
    std::vector<Individual> &individuals);

void hoang_reassign_weights(
    Population &population,
    Scalarizer &scalarizer,
    size_t dim,
    std::vector<Individual> &individuals);

void hoang_scalarize_index(Population &population, ScalarizerIndex &index, std::vector<Individual> &individuals);
void assign_scalarization_weights_if_necessary(Population &population, std::vector<Individual> &individuals);