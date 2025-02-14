//  DAEDALUS – Distributed and Automated Evolutionary Deep Architecture Learning with Unprecedented Scalability
// 
// This research code was developed as part of the research programme Open Technology Programme with project number 18373, which was financed by the Dutch Research Council (NWO), Elekta, and Ortec Logiqcare.
// 
// Project leaders: Peter A.N. Bosman, Tanja Alderliesten
// Researchers: Alex Chebykin, Arthur Guijt, Vangelis Kostoulas
// Main code developer: Arthur Guijt

#include "scalarize.hpp"

Matrix<double> getRandomAbsSumNormalizedVectors(Rng &rng, size_t num_vectors, size_t dim)
{
    std::uniform_real_distribution<double> r01(0.0, 1.0);
    // Note: matrix is row major, so having the dimension itself be sequential is preferred.
    Matrix<double> vectors(0, dim, num_vectors);
    for (size_t row = 0; row < num_vectors; ++row)
    {
        double abs_sum = 0.0;
        for (size_t col = 0; col < dim; ++col)
        {
            double v = r01(rng.rng);
            abs_sum += v;
            vectors[{row, col}] = v;
        }
        for (size_t col = 0; col < dim; ++col)
        {
            vectors[{row, col}] /= abs_sum;
        }
    }

    return vectors;
}

int is_any_individual_scalarizationweight_uninit(
    TypedGetter<ScalarizationWeights> &sw,
    std::vector<Individual> &individuals,
    size_t scalarizer_id,
    size_t dim
)
{
    int r = 0;
    for (auto &p : individuals)
    {
        auto &csw = sw.getData(p);
        if (csw.weights.size() <= scalarizer_id) {
            r = 1;
            break;
        }
        if (csw.weights[scalarizer_id].size() != dim) {
            r = 2;
            break;
        }
    }
    return r;
}

void hoang_assign_weights(
    Population &population,
    Scalarizer &scalarizer,
    std::vector<Individual> &individuals,
    Matrix<double> &vectors,
    std::vector<size_t> &vector_indices,
    size_t dim)
{
    auto sw = population.getDataContainer<ScalarizationWeights>();
    size_t sc_id = *scalarizer.scalarizer_id;
    // First - ensure there are sufficiently many weights.
    for (auto &ind : individuals) {
        auto &d = sw.getData(ind);
        if (d.weights.size() <= sc_id) {
            d.weights.resize(sc_id + 1);
        }
    }
    int any_uninit = is_any_individual_scalarizationweight_uninit(sw, individuals, sc_id, dim);
    t_assert(any_uninit != 1, "after the first loop any uninit should be 0 or 2");

    std::vector<size_t> leftover(individuals.size());
    std::iota(leftover.begin(), leftover.end(), 0);
    for (size_t vii_idx = 0; vii_idx < vector_indices.size(); ++vii_idx)
    {
        size_t v_idx = vector_indices[vii_idx];
        double *v_start = &vectors[{v_idx, 0}];
        size_t best_l_idx = 0;
        // Just like single-objective, scalarization is better if lower
        double best_scalarization = std::numeric_limits<double>::infinity();
        for (size_t l_idx = 0; l_idx < leftover.size(); ++l_idx)
        {
            size_t i_idx = leftover[l_idx];
            auto &ii = individuals[i_idx];
            // Set scalarization
            auto &sc = sw.getData(ii);
            sc.weights[sc_id].resize(dim);
            std::copy(v_start, v_start + dim, sc.weights[sc_id].data());
            // Test scalarization
            double scalarization = scalarizer.scalarize(ii);
            if (scalarization < best_scalarization)
            {
                best_scalarization = scalarization;
                best_l_idx = l_idx;
            }
        }
        // Stop the best individual from having its weights reassigned.
        std::swap(leftover[best_l_idx], leftover.back());
        leftover.pop_back();
    }
}

void hoang_generate_initial_weights(
    Population &population,
    Scalarizer &scalarizer,
    size_t dim,
    std::vector<Individual> &individuals
)
{
    size_t num_random_vectors = std::max(10 * individuals.size(), dim);
    Rng &rng = *population.getGlobalData<Rng>();

    // Make scalarizer population aware (i.e. min, max...)
    scalarizer.update_frame_of_reference(individuals);

    // Generate random vectors
    auto random_vectors = getRandomAbsSumNormalizedVectors(rng, num_random_vectors, dim);
    // Note: overwrite the first dim vectors with single-objective directions.
    for (size_t d = 0; d < dim; ++d)
    {
        for (size_t d2 = 0; d2 < dim; ++d2)
        {
            random_vectors[{d, d2}] = 0.0;
        }
        random_vectors[{d, d}] = 1.0;
    }

    // Select well distributed subsample
    auto distance = [&random_vectors, dim](size_t i, size_t j) {
        // Note: this makes use of the fact that our Matrix is row-major
        //       and that each vector's elements are laid out in rows.
        return euclidean_distance(&random_vectors[{i, 0}], &random_vectors[{j, 0}], dim);
    };
    auto indices_of_interest = greedyScatteredSubsetSelection(distance, num_random_vectors, individuals.size(), 0);

    // TODO: Replace this with a simpler assignment method, as the actual assignment is overriden by the scheme below.
    hoang_assign_weights(population, scalarizer, individuals, random_vectors, indices_of_interest, dim);
}

void hoang_reassign_weights(
    Population &population,
    Scalarizer &scalarizer,
    size_t dim,
    std::vector<Individual> &individuals
)
{
    scalarizer.update_frame_of_reference(individuals);
    size_t sc_id = *scalarizer.scalarizer_id;

    size_t num_vectors = individuals.size();
    Matrix<double> vectors(0.0, dim, num_vectors);
    Rng &rng = *population.getGlobalData<Rng>();

    // Gather weights
    TypedGetter<ScalarizationWeights> tgsw = population.getDataContainer<ScalarizationWeights>();
    for (size_t idx = 0; idx < individuals.size(); ++idx)
    {
        Individual &ii = individuals[idx];
        auto sc = tgsw.getData(ii);
        std::copy(sc.weights[sc_id].begin(), sc.weights[sc_id].end(), &vectors[{idx, 0}]);
    }
    // Shuffle the indices
    std::vector<size_t> indices(individuals.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), rng.rng);
    // Assign weights
    hoang_assign_weights(population, scalarizer, individuals, vectors, indices, dim);
}

void hoang_scalarize_index(Population &population, ScalarizerIndex &index, std::vector<Individual> &individuals)
{
    auto sw = population.getDataContainer<ScalarizationWeights>();
    // For each scalarizer present.
    for (size_t i = 0; i < index.scalarizers.size(); ++i)
    {
        auto &scalarizer = *index.scalarizers[i];
        size_t scalarizer_dim = scalarizer.get_dim();
        // Check if there any solutions without weights assigned.
        int any_uninit = is_any_individual_scalarizationweight_uninit(sw, individuals, i, scalarizer_dim);

        // If there are - generate new weights
        if (any_uninit > 0)
        {
            hoang_generate_initial_weights(population, scalarizer, scalarizer.get_dim(), individuals);
            // any_uninit = is_any_individual_scalarizationweight_uninit(sw, individuals, i, scalarizer_dim);
            // t_assert(any_uninit == 0, "After assigning initial weights, we should not have this problem.");
        }
        else
        {
            hoang_reassign_weights(population, scalarizer, scalarizer.get_dim(), individuals);
            // any_uninit = is_any_individual_scalarizationweight_uninit(sw, individuals, i, scalarizer_dim);
            // t_assert(any_uninit == 0, "After assigning initial weights, we should not have this problem.");
        }
    }
}

void assign_scalarization_weights_if_necessary(
    Population &population,
    std::vector<Individual> &individuals
)
{
    // If there is no ScalarizerIndex - there is nothing to scalarize for.
    if (! population.isGlobalRegistered<ScalarizerIndex>()) {
        return;
    }
    
    ScalarizerIndex& index = *population.getGlobalData<ScalarizerIndex>();
    hoang_scalarize_index(population, index, individuals);
}