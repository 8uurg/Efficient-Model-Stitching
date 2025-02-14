//  DAEDALUS – Distributed and Automated Evolutionary Deep Architecture Learning with Unprecedented Scalability
// 
// This research code was developed as part of the research programme Open Technology Programme with project number 18373, which was financed by the Dutch Research Council (NWO), Elekta, and Ortec Logiqcare.
// 
// Project leaders: Peter A.N. Bosman, Tanja Alderliesten
// Researchers: Alex Chebykin, Arthur Guijt, Vangelis Kostoulas
// Main code developer: Arthur Guijt

#pragma once

#include "base.hpp"
#include <filesystem>

// Common
class missing_file : public std::exception
{
  public:
    missing_file(){};

    const char *what() const throw()
    {
        return "file is missing";
    }
};

class invalid_instance : public std::exception
{
  public:
    invalid_instance(){};

    const char *what() const throw()
    {
        return "file does not contain a valid instance";
    }
};

// Generic Problems

/**
 * Evaluation function taking discrete input
 */
class DiscreteObjectiveFunction : public ObjectiveFunction
{
    std::function<double(std::vector<char> &)> evaluation_function;
    size_t l;
    std::vector<char> alphabet_size;
    size_t index;

  public:
    DiscreteObjectiveFunction(std::function<double(std::vector<char> &)> evaluation_function,
                              size_t l,
                              std::vector<char> alphabet_size,
                              size_t index = 0);

    // Convinience method: note that alphabet_size is converted to std::vector<char>.
    // As char is often seen as a string-related method, a version with a numeric type
    // can be convinient is some cases.
    DiscreteObjectiveFunction(std::function<double(std::vector<char> &)> evaluation_function,
                              size_t l,
                              std::vector<short> alphabet_size,
                              size_t index = 0);
    void evaluate(Individual i) override;

    ObjectiveFunction *clone() override;

    void registerData() override;

    std::function<double(std::vector<char> &)> get_evaluation_function()
    {
        return evaluation_function;
    }
    size_t get_l()
    {
        return l;
    }
    std::vector<char> get_alphabet_size()
    {
        return alphabet_size;
    }
    size_t get_index()
    {
        return index;
    }
};

/**
 * Evaluation function taking continuous input
 */
class ContinuousObjectiveFunction : public ObjectiveFunction
{
    std::function<double(std::vector<double> &)> evaluation_function;
    size_t l;
    size_t index;

  public:
    ContinuousObjectiveFunction(std::function<double(std::vector<double> &)>, size_t l, size_t index = 0);
    void evaluate(Individual i) override;

    ObjectiveFunction *clone() override;

    void registerData() override;

    std::function<double(std::vector<double> &)> get_evaluation_function()
    {
        return evaluation_function;
    }
    size_t get_l()
    {
        return l;
    }
    size_t get_index()
    {
        return index;
    }
};

// Problem: OneMax

// Common functions
double evaluate_onemax(size_t l, std::vector<char> &genotype);

// OneMax
//
// A simple sum over binary variables for a string length of l.
class OneMax : public ObjectiveFunction
{
    size_t l;
    size_t index;

    struct Cache
    {
        TypedGetter<GenotypeCategorical> gc;
        TypedGetter<Objective> go;
    };
    std::optional<Cache> cache;

  public:
    OneMax(size_t l, size_t index = 0);

    void evaluate(Individual i) override;

    ObjectiveFunction *clone() override;

    void registerData() override;
    void afterRegisterData() override;

    size_t get_l()
    {
        return l;
    }
    size_t get_index()
    {
        return index;
    }
};

class ZeroMax : public ObjectiveFunction
{
    size_t l;
    size_t index;

    struct Cache
    {
        TypedGetter<GenotypeCategorical> gc;
        TypedGetter<Objective> go;
    };
    std::optional<Cache> cache;

  public:
    ZeroMax(size_t l, size_t index = 0);

    void evaluate(Individual i) override;

    ObjectiveFunction *clone() override;

    void registerData() override;
    void afterRegisterData() override;

    size_t get_l()
    {
        return l;
    }
    size_t get_index()
    {
        return index;
    }
};

// Problem: MaxCut

// A weighted edge
struct Edge
{
    size_t i;
    size_t j;
    double w;
};

// A MaxCut instance consisting of weighted edges.
struct MaxCutInstance
{
    size_t num_vertices;
    size_t num_edges;
    std::vector<Edge> edges;
};

// Common functions
double evaluate_maxcut(MaxCutInstance &instance, std::vector<char> &genotype);
MaxCutInstance load_maxcut(std::istream &in);
MaxCutInstance load_maxcut(std::filesystem::path instancePath);

class MaxCut : public ObjectiveFunction
{
    MaxCutInstance instance;
    size_t index;

    struct Cache
    {
        TypedGetter<GenotypeCategorical> ggc;
        TypedGetter<Objective> go;
    };
    std::optional<Cache> cache;

  public:
    MaxCut(MaxCutInstance edges, size_t index = 0);
    MaxCut(std::filesystem::path &path, size_t index = 0);

    void evaluate(Individual i) override;

    ObjectiveFunction *clone() override;

    void registerData() override;

    MaxCutInstance get_instance()
    {
        return instance;
    }

    size_t get_index()
    {
        return index;
    }
};

// Problem: Best-of-Traps

//
struct ConcatenatedPermutedTrap
{
    size_t number_of_parameters;
    size_t block_size;
    std::vector<size_t> permutation;
    std::vector<char> optimum;
};

struct BestOfTrapsInstance
{
    size_t l;
    std::vector<ConcatenatedPermutedTrap> concatenatedPermutedTraps;
};

// Common functions
int evaluate_BestOfTraps(BestOfTrapsInstance &bestOfTraps, char *solution, size_t &best_fn);
BestOfTrapsInstance load_BestOfTraps(std::filesystem::path inpath);
BestOfTrapsInstance load_BestOfTraps(std::istream &in);

class BestOfTraps : public ObjectiveFunction
{
    BestOfTrapsInstance instance;
    size_t index;

    struct Cache
    {
        TypedGetter<GenotypeCategorical> ggc;
        TypedGetter<Objective> go;
    };
    std::optional<Cache> cache;

  public:
    BestOfTraps(BestOfTrapsInstance instance, size_t index = 0);
    BestOfTraps(std::filesystem::path &path, size_t index = 0);

    void evaluate(Individual i) override;

    ObjectiveFunction *clone() override;

    void registerData() override;

    BestOfTrapsInstance get_instance()
    {
        return instance;
    }
    size_t get_index()
    {
        return index;
    }
    
};

class WorstOfTraps : public ObjectiveFunction
{
    BestOfTrapsInstance instance;
    size_t index;

    struct Cache
    {
        TypedGetter<GenotypeCategorical> ggc;
        TypedGetter<Objective> go;
    };
    std::optional<Cache> cache;

  public:
    WorstOfTraps(BestOfTrapsInstance instance, size_t index = 0);
    WorstOfTraps(std::filesystem::path &path, size_t index = 0);

    void evaluate(Individual i) override;

    ObjectiveFunction *clone() override;

    void registerData() override;

    BestOfTrapsInstance get_instance()
    {
        return instance;
    }
    size_t get_index()
    {
        return index;
    }
};

// Compose multiple functions together by calling them sequentially.
//
// Tip: Force the different functions to target different objectives to construct a multi-objective function
//      from single-objective functions.
class Compose : public ObjectiveFunction
{
    std::vector<std::shared_ptr<ObjectiveFunction>> problems;

  public:
    Compose(std::vector<std::shared_ptr<ObjectiveFunction>> problems) : problems(problems)
    {
    }
    void setPopulation(std::shared_ptr<Population> population) override;
    void registerData() override;
    void afterRegisterData() override;

    void evaluate(Individual i) override;

    ObjectiveFunction *clone() override;

    std::vector<std::shared_ptr<ObjectiveFunction>> get_problems()
    {
        return problems;
    }

};

// Peter's Benchmark Function
double GOMEA_HierarchicalDeceptiveTrapProblemEvaluation(int l, int k, char *genes);

class HierarchicalDeceptiveTrap : public ObjectiveFunction
{
    size_t l;
    size_t k;
    size_t index;

    struct Cache
    {
        TypedGetter<GenotypeCategorical> ggc;
        TypedGetter<Objective> go;
    };
    std::optional<Cache> cache;

  public:
    HierarchicalDeceptiveTrap(size_t l, size_t k = 3, size_t index = 0);
    void evaluate(Individual ii) override;

    ObjectiveFunction *clone() override;

    void registerData() override;

    
    size_t get_l()
    {
        return l;
    }
    size_t get_k()
    {
        return k;
    }
    size_t get_index()
    {
        return index;
    }
};

//
struct NKSubfunction
{
    std::vector<size_t> variables;
    std::vector<double> lut;
};

struct NKLandscapeInstance
{
    std::vector<NKSubfunction> subfunctions;
    size_t l;
};

NKLandscapeInstance load_nklandscape(std::istream &in);
NKLandscapeInstance load_nklandscape(std::filesystem::path instancePath);
double evaluate_nklandscape(NKLandscapeInstance &instance, std::vector<char> &genotype);

class NKLandscape : public ObjectiveFunction
{
    NKLandscapeInstance instance;
    size_t index;

    struct Cache
    {
        TypedGetter<GenotypeCategorical> ggc;
        TypedGetter<Objective> go;
    };
    std::optional<Cache> cache;

    void doCache();

  public:
    NKLandscape(std::filesystem::path path, size_t index = 0);
    NKLandscape(NKLandscapeInstance instance, size_t index = 0);
    void evaluate(Individual ii) override;

    ObjectiveFunction *clone() override;

    void registerData() override;

    NKLandscapeInstance get_instance()
    {
        return instance;
    }

    size_t get_index()
    {
        return index;
    }
};