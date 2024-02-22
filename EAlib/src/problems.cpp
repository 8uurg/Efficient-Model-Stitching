#include "problems.hpp"
#include "base.hpp"
#include "cppassert.h"
#include <fstream>
#include <limits>

// General purpose helpers.
void register_discrete_data(Population &population, size_t l, std::vector<char> alphabet_size)
{
    population.registerData<Objective>();
    population.registerData<GenotypeCategorical>();
    if (!population.isGlobalRegistered<GenotypeCategoricalData>())
    {
        population.registerGlobalData(GenotypeCategoricalData {l, alphabet_size});
    }
    else
    {
        // Update data instead -- alphabets must match however
        auto &categorical_spec = *population.getGlobalData<GenotypeCategoricalData>();
        // Alphabet must match up to a shared point, and l for this problem is larger, be replaced.
        for (size_t a_i = 0; a_i < categorical_spec.l && a_i < l; ++a_i)
        {
            t_assert(categorical_spec.alphabet_size[a_i] == alphabet_size[a_i], "Alphabet sizes must match");
        }
        if (categorical_spec.l < l)
        {
            categorical_spec.alphabet_size = alphabet_size;
        }

        // String length should be the maximum of the two
        categorical_spec.l = std::max(l, categorical_spec.l);
    }
}

// Generic Evaluation Functions

DiscreteObjectiveFunction::DiscreteObjectiveFunction(std::function<double(std::vector<char> &)> evaluation_function,
                                                     size_t l,
                                                     std::vector<char> alphabet_size,
                                                     size_t index) :
    evaluation_function(evaluation_function), l(l), alphabet_size(alphabet_size), index(index)
{
}

std::vector<char> narrow_vec_short_to_char(std::vector<short> s)
{
    std::vector<char> c(s.size());
    for (size_t i = 0; i < s.size(); ++i)
    {
        c[i] = static_cast<char>(s[i]);
    }
    return c;
}

DiscreteObjectiveFunction::DiscreteObjectiveFunction(std::function<double(std::vector<char> &)> evaluation_function,
                                                     size_t l,
                                                     std::vector<short> alphabet_size,
                                                     size_t index) :
    evaluation_function(evaluation_function), l(l), alphabet_size(narrow_vec_short_to_char(alphabet_size)), index(index)
{
}

void DiscreteObjectiveFunction::registerData()
{
    Population &population = *(this->population);
    population.registerData<Objective>();
    population.registerData<GenotypeCategorical>();
    population.registerGlobalData(GenotypeCategoricalData { l, alphabet_size });
    // population.registerGlobalData(GObjectiveFunction(static_cast<ObjectiveFunction *>(this)));
}

void DiscreteObjectiveFunction::evaluate(Individual i)
{
    Population &population = *(this->population);
    GenotypeCategorical &genotype = population.getData<GenotypeCategorical>(i);
    t_assert(genotype.genotype.size() == l,
             "Genotype exactly the specified size, is the initial solution generator broken?");
    Objective &objective = population.getData<Objective>(i);
    if (objective.objectives.size() <= index)
        objective.objectives.resize(index + 1);
    objective.objectives[index] = evaluation_function(genotype.genotype);
}

ObjectiveFunction *DiscreteObjectiveFunction::clone()
{
    return new DiscreteObjectiveFunction(evaluation_function, l, alphabet_size, index);
}

ContinuousObjectiveFunction::ContinuousObjectiveFunction(
    std::function<double(std::vector<double> &)> evaluation_function, size_t l, size_t index) :
    evaluation_function(evaluation_function), l(l), index(index)
{
}

void ContinuousObjectiveFunction::registerData()
{
    Population &population = *(this->population);
    population.registerData<Objective>();
    population.registerData<GenotypeContinuous>();
    population.registerGlobalData(GenotypeContinuousLength { l });
    // population.registerGlobalData(GObjectiveFunction(static_cast<ObjectiveFunction *>(this)));
}

void ContinuousObjectiveFunction::evaluate(Individual i)
{
    Population &population = *(this->population);
    GenotypeContinuous &genotype = population.getData<GenotypeContinuous>(i);
    t_assert(genotype.genotype.size() == l,
             "Genotype should be exactly the specified size, is the initial solution generator broken?");
    Objective &objective = population.getData<Objective>(i);

    if (objective.objectives.size() <= index)
        objective.objectives.resize(index + 1);
    objective.objectives[index] = evaluation_function(genotype.genotype);
}

ObjectiveFunction *ContinuousObjectiveFunction::clone()
{
    return new ContinuousObjectiveFunction(evaluation_function, l, index);
}

// Problem: OneMax

double evaluate_onemax(size_t l, std::vector<char> &genotype)
{
    t_assert(genotype.size() >= l, "Genotype should be at least as large as the underlying genotype.");

    int count = 0;
    for (char g : genotype)
        count += g;
    return count;
}

OneMax::OneMax(size_t l, size_t index) : l(l), index(index)
{
}

void OneMax::evaluate(Individual ii)
{
    if (!cache.has_value())
    {
        Population &population = *(this->population);

        auto go = population.getDataContainer<Objective>();
        auto gg = population.getDataContainer<GenotypeCategorical>();
        cache.emplace(Cache{gg, go});
    }

    GenotypeCategorical &genotype = cache->gc.getData(ii);
    Objective &objective = cache->go.getData(ii);
    if (objective.objectives.size() <= index)
        objective.objectives.resize(index + 1);
    // Negate has BestOfTraps has higher = better, and Objective assumes lower = better.
    objective.objectives[index] = -evaluate_onemax(l, genotype.genotype);
}

void OneMax::registerData()
{
    std::vector<char> alphabet_size(l);
    std::fill(alphabet_size.begin(), alphabet_size.end(), 2);

    Population &population = *(this->population);
    register_discrete_data(population, l, alphabet_size);
}
void OneMax::afterRegisterData()
{
    // Population &population = *(this->population);
    // t_assert(population.isRegistered<GenotypeCategorical>(), "Genotype is absent, has an initializer been
    // registered?");
}

ObjectiveFunction *OneMax::clone()
{
    return new OneMax(l, index);
}

double evaluate_zeromax(size_t l, std::vector<char> &genotype)
{
    t_assert(genotype.size() >= l, "Genotype should be at least as large as the underlying genotype.");

    int count = 0;
    for (char g : genotype)
        count += (1 - g);
    return count;
}

ZeroMax::ZeroMax(size_t l, size_t index) : l(l), index(index)
{
}

void ZeroMax::evaluate(Individual ii)
{
    if (!cache.has_value())
    {
        Population &population = *(this->population);

        auto go = population.getDataContainer<Objective>();
        auto gg = population.getDataContainer<GenotypeCategorical>();
        cache.emplace(Cache{gg, go});
    }

    GenotypeCategorical &genotype = cache->gc.getData(ii);
    Objective &objective = cache->go.getData(ii);
    if (objective.objectives.size() <= index)
        objective.objectives.resize(index + 1);
    // Negate has BestOfTraps has higher = better, and Objective assumes lower = better.
    objective.objectives[index] = -evaluate_zeromax(l, genotype.genotype);
}

void ZeroMax::registerData()
{
    std::vector<char> alphabet_size(l);
    std::fill(alphabet_size.begin(), alphabet_size.end(), 2);

    Population &population = *(this->population);
    register_discrete_data(population, l, alphabet_size);
}
void ZeroMax::afterRegisterData()
{
    // Population &population = *(this->population);
    // t_assert(population.isRegistered<GenotypeCategorical>(), "Genotype is absent, has an initializer been
    // registered?");
}

ObjectiveFunction *ZeroMax::clone()
{
    return new ZeroMax(l, index);
}

// Problem: MaxCut

//
double evaluate_maxcut(MaxCutInstance &instance, std::vector<char> &genotype)
{
    double weight = 0;
    for (Edge edge : instance.edges)
    {
        size_t from = edge.i;
        size_t to = edge.j;
        double w = edge.w;

        weight += w * (genotype[from] != genotype[to]);
    }
    return weight;
}

MaxCutInstance load_maxcut(std::istream &in)
{
    MaxCutInstance instance;
    in >> instance.num_vertices >> instance.num_edges;

    instance.edges.resize(instance.num_edges);

    for (size_t e = 0; e < instance.num_edges; ++e)
    {
        size_t from;
        size_t to;
        double weight;

        in >> from >> to >> weight;

        if (in.fail())
            throw invalid_instance();

        instance.edges[e] = Edge{from - 1, to - 1, weight};
    }
    return instance;
}

MaxCutInstance load_maxcut(std::filesystem::path instancePath)
{
    if (!std::filesystem::exists(instancePath))
        throw missing_file();

    std::ifstream in(instancePath);

    MaxCutInstance instance = load_maxcut(in);

    in.close();

    return instance;
}

MaxCut::MaxCut(MaxCutInstance instance, size_t index) : instance(std::move(instance)), index(index)
{
}

MaxCut::MaxCut(std::filesystem::path &path, size_t index) : index(index)
{
    instance = load_maxcut(path);
}
void MaxCut::evaluate(Individual ii)
{
    if (!cache.has_value())
    {
        Population &population = *(this->population);

        auto go = population.getDataContainer<Objective>();
        auto ggc = population.getDataContainer<GenotypeCategorical>();
        cache.emplace(Cache{ggc, go});
    }

    GenotypeCategorical &genotype = cache->ggc.getData(ii);
    Objective &objective = cache->go.getData(ii);
    if (objective.objectives.size() <= index)
        objective.objectives.resize(index + 1);
    // Negate has BestOfTraps has higher = better, and Objective assumes lower = better.
    objective.objectives[index] = -evaluate_maxcut(instance, genotype.genotype);
}

void MaxCut::registerData()
{
    size_t l = instance.num_vertices;
    std::vector<char> alphabet_size(l);
    std::fill(alphabet_size.begin(), alphabet_size.end(), 2);

    Population &population = *(this->population);
    register_discrete_data(population, l, alphabet_size);
}

ObjectiveFunction *MaxCut::clone()
{
    return new MaxCut(instance, index);
}

// Problem: Best-of-Traps

//
void stopInvalidInstanceBOT(std::istream & /* stream */, std::string /* expected */)
{
    // std::cerr << "Instance provided for BOT is invalid.\n";
    // std::cerr << "Invalid character at position " << stream.tellg() << ".\n";
    // std::cerr << expected << std::endl;
    throw invalid_instance();
}
//
void stopFileMissingBOT(std::filesystem::path /* file */)
{
    // std::cerr << "Instance provided for BOT is invalid.\n";
    // std::cerr << "File " << file << " does not exist." << std::endl;
    throw missing_file();
}

BestOfTrapsInstance load_BestOfTraps(std::istream &file)
{
    size_t number_of_subfunctions = 0;
    file >> number_of_subfunctions;
    if (file.fail())
        stopInvalidInstanceBOT(file, "expected number_of_subfunctions");

    std::vector<ConcatenatedPermutedTrap> subfunctions;
    subfunctions.reserve(number_of_subfunctions);

    size_t l = 0;

    for (int fn = 0; fn < static_cast<int>(number_of_subfunctions); ++fn)
    {
        size_t number_of_parameters = 0;
        size_t block_size = 0;
        file >> number_of_parameters >> block_size;
        if (file.fail())
            throw invalid_instance();
        std::string current_line;
        // Skip to the next line.
        if (!std::getline(file, current_line))
            throw invalid_instance();
        // optimum
        std::vector<char> optimum;
        optimum.reserve(number_of_parameters);
        if (!std::getline(file, current_line))
            stopInvalidInstanceBOT(file, "expected optimum");

        {
            std::stringstream linestream(current_line);
            int v = 0;
            while (!linestream.fail())
            {
                linestream >> v;
                optimum.push_back(static_cast<char>(v));
            }
        }
        std::vector<size_t> permutation;
        permutation.reserve(number_of_parameters);
        if (!std::getline(file, current_line))
            stopInvalidInstanceBOT(file, "expected permutation");
        {
            std::stringstream linestream(current_line);
            int v = 0;
            while (!linestream.fail())
            {
                linestream >> v;
                permutation.push_back(v);
            }
        }

        l = std::max(l, (size_t)number_of_parameters);

        ConcatenatedPermutedTrap prt = ConcatenatedPermutedTrap{number_of_parameters, block_size, permutation, optimum};

        subfunctions.push_back(prt);
    }
    return BestOfTrapsInstance{l, subfunctions};
}

BestOfTrapsInstance load_BestOfTraps(std::filesystem::path inpath)
{
    if (!std::filesystem::exists(inpath))
        throw missing_file();
    std::ifstream file(inpath);

    BestOfTrapsInstance instance = load_BestOfTraps(file);

    file.close();

    return instance;
}

// The deceptive trap function
//
// With a deceptive attractor at unitation = 0
// And the optimum at unitation = size
int trapFunction(int unitation, int size)
{
    if (unitation == size)
        return size;
    return size - unitation - 1;
}

// Evaluate a single concatenated permuted trap (i.e., one of the Best-of-Traps subfunctions).
int evaluateConcatenatedPermutedTrap(ConcatenatedPermutedTrap &concatenatedPermutedTrap, char *solution)
{
    size_t l = concatenatedPermutedTrap.number_of_parameters;
    // int number_of_blocks = l / block_size;

    int objective = 0;
    for (size_t block_start = 0; block_start < l; block_start += concatenatedPermutedTrap.block_size)
    {
        int unitation = 0;
        size_t current_block_size = std::min(concatenatedPermutedTrap.block_size, l - block_start);
        for (size_t i = 0; i < current_block_size; ++i)
        {
            size_t idx = concatenatedPermutedTrap.permutation[block_start + i];
            unitation += solution[idx] == concatenatedPermutedTrap.optimum[idx];
        }
        objective += trapFunction(unitation, (int)current_block_size);
    }
    return objective;
}

int evaluate_BestOfTraps(BestOfTrapsInstance &bestOfTraps, char *solution, size_t &best_fn)
{
    int result = std::numeric_limits<int>::lowest();
    best_fn = -1;
    for (size_t fn = 0; fn < bestOfTraps.concatenatedPermutedTraps.size(); ++fn)
    {
        int result_subfn = evaluateConcatenatedPermutedTrap(bestOfTraps.concatenatedPermutedTraps[fn], solution);
        if (result_subfn > result)
        {
            best_fn = fn;
            result = result_subfn;
        }
    }
    return result;
}

BestOfTraps::BestOfTraps(BestOfTrapsInstance instance, size_t index) : instance(std::move(instance)), index(index)
{
}

BestOfTraps::BestOfTraps(std::filesystem::path &path, size_t index) : index(index)
{
    instance = load_BestOfTraps(path);
}
void BestOfTraps::evaluate(Individual ii)
{
    if (!cache.has_value())
    {
        Population &population = *(this->population);

        auto go = population.getDataContainer<Objective>();
        auto ggc = population.getDataContainer<GenotypeCategorical>();
        cache.emplace(Cache{ggc, go});
    }

    GenotypeCategorical &genotype = cache->ggc.getData(ii);
    Objective &objective = cache->go.getData(ii);

    size_t best_fn = 0;

    if (objective.objectives.size() <= index)
        objective.objectives.resize(index + 1);
    // Negate has BestOfTraps has higher = better, and Objective assumes lower = better.
    objective.objectives[index] = -evaluate_BestOfTraps(instance, genotype.genotype.data(), best_fn);
}

void BestOfTraps::registerData()
{
    size_t l = instance.l;
    std::vector<char> alphabet_size(l);
    std::fill(alphabet_size.begin(), alphabet_size.end(), 2);

    Population &population = *(this->population);
    register_discrete_data(population, l, alphabet_size);

    // Register subfunction data
    if (!population.isGlobalRegistered<Subfunctions>())
    {
        Subfunctions subfns_data;
        population.registerGlobalData(subfns_data);
    }
    Subfunctions* subfns_data = population.getGlobalData<Subfunctions>().get();
    for (auto& subfn : instance.concatenatedPermutedTraps)
    {
        for (size_t i = 0; i < subfn.number_of_parameters; i += subfn.block_size)
        {
            size_t endpoint_excl = std::min(i + subfn.block_size, subfn.number_of_parameters);
            size_t current_block_size = endpoint_excl - i;
            std::vector<size_t> subsubfn(current_block_size);
            std::copy(subfn.permutation.begin() + static_cast<long int>(i), subfn.permutation.begin() + static_cast<long int>(endpoint_excl), subsubfn.begin());
            subfns_data->subfunctions.push_back(subsubfn);
        }
    }
}

int evaluate_WorstOfTraps(BestOfTrapsInstance &bestOfTraps, char *solution, size_t &worst_fn)
{
    int result = std::numeric_limits<int>::max();
    worst_fn = -1;
    for (size_t fn = 0; fn < bestOfTraps.concatenatedPermutedTraps.size(); ++fn)
    {
        int result_subfn = evaluateConcatenatedPermutedTrap(bestOfTraps.concatenatedPermutedTraps[fn], solution);
        if (result_subfn < result)
        {
            worst_fn = fn;
            result = result_subfn;
        }
    }
    return result;
}

ObjectiveFunction *BestOfTraps::clone()
{
    return new BestOfTraps(instance, index);
}

WorstOfTraps::WorstOfTraps(BestOfTrapsInstance instance, size_t index) : instance(std::move(instance)), index(index)
{
}

WorstOfTraps::WorstOfTraps(std::filesystem::path &path, size_t index) : index(index)
{
    instance = load_BestOfTraps(path);
}
void WorstOfTraps::evaluate(Individual ii)
{
    if (!cache.has_value())
    {
        Population &population = *(this->population);

        auto go = population.getDataContainer<Objective>();
        auto ggc = population.getDataContainer<GenotypeCategorical>();
        cache.emplace(Cache{ggc, go});
    }

    GenotypeCategorical &genotype = cache->ggc.getData(ii);
    Objective &objective = cache->go.getData(ii);

    size_t best_fn = 0;

    if (objective.objectives.size() <= index)
        objective.objectives.resize(index + 1);
    objective.objectives[index] = -evaluate_WorstOfTraps(instance, genotype.genotype.data(), best_fn);
}

void WorstOfTraps::registerData()
{
    size_t l = instance.l;
    std::vector<char> alphabet_size(l);
    std::fill(alphabet_size.begin(), alphabet_size.end(), 2);

    Population &population = *(this->population);

    register_discrete_data(population, l, alphabet_size);
}

ObjectiveFunction *WorstOfTraps::clone()
{
    return new WorstOfTraps(instance, index);
}

// Compose
void Compose::setPopulation(std::shared_ptr<Population> population)
{
    for (auto &p : problems)
    {
        p->setPopulation(population);
    }
}
void Compose::registerData()
{
    for (auto &p : problems)
    {
        p->registerData();
    }
}
void Compose::afterRegisterData()
{
    for (auto &p : problems)
    {
        p->afterRegisterData();
    }
}
void Compose::evaluate(Individual i)
{
    for (auto &p : problems)
    {
        p->evaluate(i);
    }
}

ObjectiveFunction *Compose::clone()
{
    std::vector<std::shared_ptr<ObjectiveFunction>> cloned_problems(problems.size());
    for (size_t idx = 0; idx < problems.size(); ++idx)
    {
        cloned_problems[idx] = std::shared_ptr<ObjectiveFunction>(problems[idx]->clone());
    }
    return new Compose(std::move(cloned_problems));
}

// Peter's benchmark function
double GOMEA_HierarchicalDeceptiveTrapProblemEvaluation(int l, int k, char *genes)
{
    char *symbols;
    int i, j, number_of_symbols, u;
    double result, level_result, level_multiplier;

    t_assert((l % k) == 0, "l % k should be 0");

    number_of_symbols = l;
    symbols = (char *)malloc(number_of_symbols * sizeof(char));

    for (i = 0; i < l; i++)
        symbols[i] = genes[i] == 1 ? '1' : '0';

    result = 0;
    level_multiplier = k;
    while (number_of_symbols >= k)
    {
        if ((number_of_symbols % k) != 0)
        {
            printf("Error in evaluating hierarchical deceptive trap k=%d: Number of genes is not a hierarchical "
                   "multiple of %d.\n",
                   k,
                   k);
            exit(0);
        }

        level_result = 0;
        for (i = 0; i < number_of_symbols / k; i++)
        {
            u = 0;
            for (j = i * k; j < (i + 1) * k; j++)
            {
                if (symbols[j] == '1')
                    u++;
                if (symbols[j] == '-')
                {
                    u = -1;
                    break;
                }
            }

            if (u >= 0)
            {
                if (u == k)
                    level_result += 1.0;
                else
                    level_result += ((double)(k - 1 - u)) / ((double)k);
            }

            if (u < 0)
                symbols[i] = '-';
            if (u == 0)
                symbols[i] = '0';
            if ((u > 0) && (u < k))
                symbols[i] = '-';
            if (u == k)
                symbols[i] = '1';
        }
        result += level_result * level_multiplier;
        level_multiplier *= k;
        number_of_symbols /= k;
    }

    free(symbols);

    return result;
}
HierarchicalDeceptiveTrap::HierarchicalDeceptiveTrap(size_t l, size_t k, size_t index) : l(l), k(k), index(index)
{
}
void HierarchicalDeceptiveTrap::registerData()
{
    std::vector<char> alphabet_size(l);
    std::fill(alphabet_size.begin(), alphabet_size.end(), 2);

    Population &population = *(this->population);
    register_discrete_data(population, l, alphabet_size);
}
void HierarchicalDeceptiveTrap::evaluate(Individual ii)
{
    if (!cache.has_value())
    {
        Population &population = *this->population;

        auto go = population.getDataContainer<Objective>();
        auto ggc = population.getDataContainer<GenotypeCategorical>();
        cache.emplace(Cache{ggc, go});
    }
    GenotypeCategorical &genotype = cache->ggc.getData(ii);
    Objective &objective = cache->go.getData(ii);
    t_assert(genotype.genotype.size() == l,
             "Genotype exactly the specified size, is the initial solution generator broken?");

    double o = GOMEA_HierarchicalDeceptiveTrapProblemEvaluation(
        static_cast<int>(l), static_cast<int>(k), genotype.genotype.data());

    if (objective.objectives.size() <= index)
        objective.objectives.resize(index + 1);
    objective.objectives[index] = -o;
}

ObjectiveFunction *HierarchicalDeceptiveTrap::clone()
{
    return new HierarchicalDeceptiveTrap(l, k, index);
}

/**
 * @brief Hashing for binary vector containing chars '0' and '1'
 * 
 * @param combination 
 * @param l 
 * @return size_t 
 */
size_t hash_combination_str(char* combination, size_t l)
{
    size_t r = 0;
    for (size_t o = 0; o < l; ++o)
    {
        r = r << 1;
        r += (combination[o] == '1');
    }
    return r;
}

NKLandscapeInstance load_nklandscape(std::istream &in)
{
    size_t l = 0;
    size_t num_subfunctions = 0;
    in >> l >> num_subfunctions;

    std::vector<NKSubfunction> subfunctions;
    
    // Read subfunctions
    for (size_t idx = 0; idx < num_subfunctions; ++idx)
    {
        std::vector<size_t> variables;

        size_t num_subfn_variables = 0;
        in >> num_subfn_variables;
        for (size_t v_idx = 0; v_idx < num_subfn_variables; ++v_idx)
        {
            size_t v = 0;
            in >> v;
            variables.push_back(v);
        }
        size_t num_combinations = 1 << num_subfn_variables;
        std::vector<double> lut(num_combinations);
        std::string dummy_buffer;
        for (size_t c_idx = 0; c_idx < num_combinations; ++c_idx)
        {
            std::string combination;
            double value;
            in >> combination >> value;
            std::getline(in, dummy_buffer);
            lut[hash_combination_str(&combination.data()[1], combination.size() - 2)] = value;
        }
        subfunctions.push_back(NKSubfunction {
            std::move(variables),
            std::move(lut),
        });

        t_assert(!in.fail(), "Reading the instance should have completed successfully: failed to parse subfunction");

    }

    t_assert(!in.fail(), "Reading the instance should have completed successfully.");
    
    return NKLandscapeInstance {
        subfunctions,
        l
    };
}
NKLandscapeInstance load_nklandscape(std::filesystem::path instancePath)
{
    std::ifstream in(instancePath);

    NKLandscapeInstance instance = load_nklandscape(in);

    in.close();

    return instance;
}

/**
 * @brief Hashing for binary vector containing chars, with values 0 and 1
 * 
 * (Note: not the ascii or utf8 encoding!)
 * 
 * @param combination 
 * @param l 
 * @return size_t 
 */
size_t hash_combination_arr(char* combination, size_t* variables, size_t l)
{
    size_t r = 0;
    for (size_t o = 0; o < l; ++o)
    {
        r = r << 1;
        r += (combination[variables[o]] == 1);
    }
    return r;
}

double evaluate_nksubfunction(NKSubfunction &subfn, std::vector<char> &genotype)
{
    return subfn.lut[hash_combination_arr(genotype.data(), subfn.variables.data(), subfn.variables.size())];
}

double evaluate_nklandscape(NKLandscapeInstance &instance, std::vector<char> &genotype)
{
    double r = 0.0;
    for (auto& subfn : instance.subfunctions)
    {
        r += evaluate_nksubfunction(subfn, genotype);
    }
    return r;
}

NKLandscape::NKLandscape(NKLandscapeInstance instance, size_t index) : instance(instance), index(index)
{
}
NKLandscape::NKLandscape(std::filesystem::path path, size_t index) : instance(load_nklandscape(path)), index(index)
{
}
void NKLandscape::doCache()
{
    if (cache.has_value())
        return;
    cache.emplace(Cache{
        population->getDataContainer<GenotypeCategorical>(),
        population->getDataContainer<Objective>(),
    });
}

void NKLandscape::evaluate(Individual ii)
{
    doCache();
    Cache &c = *cache;
    auto& ob = c.go.getData(ii);
    if (ob.objectives.size() >= index)
        ob.objectives.resize(index + 1);
    ob.objectives[index] = -evaluate_nklandscape(this->instance, c.ggc.getData(ii).genotype);
}
void NKLandscape::registerData()
{
    size_t l = instance.l;
    std::vector<char> alphabet_size(l);
    std::fill(alphabet_size.begin(), alphabet_size.end(), 2);

    Population &population = *(this->population);
    register_discrete_data(population, l, alphabet_size);

    // Register subfunction data
    if (!population.isGlobalRegistered<Subfunctions>())
    {
        Subfunctions subfns_data;
        population.registerGlobalData(subfns_data);
    }
    Subfunctions* subfns_data = population.getGlobalData<Subfunctions>().get();

    for (auto& subfn : instance.subfunctions)
    {
        subfns_data->subfunctions.push_back(subfn.variables);
    }
}

ObjectiveFunction *NKLandscape::clone()
{
    return new NKLandscape(instance, index);
}
