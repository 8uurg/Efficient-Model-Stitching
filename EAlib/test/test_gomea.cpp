#include "acceptation_criteria.hpp"
#include "archive.hpp"
#include "base.hpp"
#include "gomea.hpp"
#include "mocks_base_ea.hpp"
#include "mocks_gomea.hpp"
#include "problems.hpp"
#include "initializers.hpp"
#include "scalarize.hpp"

#include <algorithm>
#include <memory>
#include <numeric>
#include <catch2/catch.hpp>
#include <catch2/trompeloeil.hpp>

TEST_CASE("Mutual Information", "[Linkage][Analysis]")
{
    std::shared_ptr<Population> pop = std::make_shared<Population>();

    // General problem setup.
    size_t l = 8;
    std::vector<char> alphabet_size(l);
    std::fill(alphabet_size.begin(), alphabet_size.end(), 2);

    pop->registerGlobalData(GenotypeCategoricalData{l, alphabet_size});

    MI mi;
    mi.setPopulation(pop);
    mi.registerData();
    pop->registerData<GenotypeCategorical>();
    mi.afterRegisterData();

    // Create reference population
    size_t population_size = 8;
    std::vector<Individual> subpopulation(population_size);
    std::generate(subpopulation.begin(), subpopulation.end(), [&pop]() { return pop->newIndividual(); });
    for (size_t i = 0; i < subpopulation.size(); ++i)
    {
        auto ii = subpopulation[i];
        GenotypeCategorical &genotype = pop->getData<GenotypeCategorical>(ii);
        genotype.genotype.resize(l);
        std::fill(genotype.genotype.begin(), genotype.genotype.end(), 0);
        // keep 0 and 1 as converged
        // 2 varies every other.
        genotype.genotype[2] = i % 2;
        // 3 is correlated (opposite)
        genotype.genotype[3] = 1 - i % 2;
        // 4 is 0, 0, 1, 1 repeating.
        genotype.genotype[4] = (i % 4) / 2;
        // 5 is equal to 4, unless 2 is equal to 0, then 5 is `1`.
        genotype.genotype[5] = genotype.genotype[2] == 0 ? 1 : genotype.genotype[4];
        // 6 only has a single 1 at the start.
        genotype.genotype[6] = i == 0 ? 1 : 0;
        // 7 only has a single 0 at the start.
        genotype.genotype[7] = i == 0 ? 0 : 1;

        //       Individual
        //     0 1 2 3 4 5 6 7
        // v_0 0 0 0 0 0 0 0 0
        // v_1 0 0 0 0 0 0 0 0
        // v_2 0 1 0 1 0 1 0 1
        // v_3 1 0 1 0 1 0 1 0
        // v_4 0 0 1 1 0 0 1 1
        // v_5 1 0 1 1 1 0 1 1
        // v_6 1 0 0 0 0 0 0 0
        // v_7 0 1 1 1 1 1 1 1
    }

    //
    SECTION("Converged variables should provide MI of 0")
    {
        // initial population consists of all zeroes
        double c_mi = mi.compute_linkage(0, 1, subpopulation);
        REQUIRE(c_mi == Approx(0.0));
    }
    SECTION("One converged variable should provide MI of 0")
    {
        // initial population consists of all zeroes
        double c_mi = mi.compute_linkage(0, 2, subpopulation);
        REQUIRE(c_mi == Approx(0.0));
    }
    SECTION("Correlated (in equal counts) variables should provide MI of 1")
    {
        double c_mi = mi.compute_linkage(2, 3, subpopulation);
        REQUIRE(c_mi == Approx(1.0));
    }
    SECTION("Uncorrelated variables should provide MI of 0")
    {
        double c_mi = mi.compute_linkage(2, 4, subpopulation);
        REQUIRE(c_mi == Approx(0.0));
    }
    SECTION("Variables that are somewhat related should provide MI between 0 and 1")
    {
        double c_mi = mi.compute_linkage(2, 5, subpopulation);
        REQUIRE((c_mi > 0.0 && c_mi < 1.0));
    }
    SECTION("Correlated (in non-equal count) variables should provide MI between 0 and 1")
    {
        double c_mi = mi.compute_linkage(6, 7, subpopulation);
        REQUIRE((c_mi > 0.0 && c_mi < 1.0));
    }
}

TEST_CASE("Mutual Information - Varying Alphabet", "[Linkage][Analysis]")
{
    std::shared_ptr<Population> pop = std::make_shared<Population>();

    // General problem setup.
    size_t l = 8;
    std::vector<char> alphabet_size(l);
    std::fill(alphabet_size.begin(), alphabet_size.end(), 2);
    // Special cases
    alphabet_size[1] = 3;
    alphabet_size[4] = 3;

    pop->registerGlobalData(GenotypeCategoricalData{l, alphabet_size});

    MI mi;
    mi.setPopulation(pop);
    mi.registerData();
    pop->registerData<GenotypeCategorical>();
    mi.afterRegisterData();

    // Create reference population
    size_t population_size = 8;
    std::vector<Individual> subpopulation(population_size);
    std::generate(subpopulation.begin(), subpopulation.end(), [&pop]() { return pop->newIndividual(); });
    for (size_t i = 0; i < subpopulation.size(); ++i)
    {
        auto ii = subpopulation[i];
        GenotypeCategorical &genotype = pop->getData<GenotypeCategorical>(ii);
        genotype.genotype.resize(l);
        std::fill(genotype.genotype.begin(), genotype.genotype.end(), 0);
        // keep 0 and 1 as converged
        // 2 varies every other.
        genotype.genotype[2] = i % 2;
        // 3 is correlated (opposite)
        genotype.genotype[3] = 1 - i % 2;
        // 4 is 0, 1, 2 repeating.
        genotype.genotype[4] = i % 3;
        // 5 is equal to 4, unless 2 is equal to 0, then 5 is `1`.
        genotype.genotype[5] = genotype.genotype[2] == 0 ? 1 : genotype.genotype[4];
        // 6 only has a single 1 at the start.
        genotype.genotype[6] = i == 0 ? 1 : 0;
        // 7 only has a single 0 at the start.
        genotype.genotype[7] = i == 0 ? 0 : 1;

        //       Individual
        //     0 1 2 3 4 5 6 7
        // v_0 0 0 0 0 0 0 0 0
        // v_1 0 0 0 0 0 0 0 0
        // v_2 0 1 0 1 0 1 0 1
        // v_3 1 0 1 0 1 0 1 0
        // v_4 0 1 2 0 1 2 0 1
        // v_5 1 0 1 1 1 0 1 1
        // v_6 1 0 0 0 0 0 0 0
        // v_7 0 1 1 1 1 1 1 1
    }

    //
    SECTION("Converged variables should provide MI of 0 - even if alphabet size differs")
    {
        // initial population consists of all zeroes
        double c_mi = mi.compute_linkage(0, 1, subpopulation);
        REQUIRE(c_mi == Approx(0.0));
    }
    SECTION("Computing MI with variables of different cardinality should work.")
    {
        double c_mi = mi.compute_linkage(2, 4, subpopulation);
        REQUIRE(c_mi >= 0.0);
    }
}

TEST_CASE("Normalized Mutual Information", "[Linkage][Analysis]")
{
    std::shared_ptr<Population> pop = std::make_shared<Population>();

    // General problem setup.
    size_t l = 8;
    std::vector<char> alphabet_size(l);
    std::fill(alphabet_size.begin(), alphabet_size.end(), 2);

    pop->registerGlobalData(GenotypeCategoricalData{l, alphabet_size});

    NMI nmi;
    nmi.setPopulation(pop);
    nmi.registerData();
    pop->registerData<GenotypeCategorical>();
    nmi.afterRegisterData();

    // Create reference population
    size_t population_size = 8;
    std::vector<Individual> subpopulation(population_size);
    std::generate(subpopulation.begin(), subpopulation.end(), [&pop]() { return pop->newIndividual(); });
    for (size_t i = 0; i < subpopulation.size(); ++i)
    {
        auto ii = subpopulation[i];
        GenotypeCategorical &genotype = pop->getData<GenotypeCategorical>(ii);
        genotype.genotype.resize(l);
        std::fill(genotype.genotype.begin(), genotype.genotype.end(), 0);
        // keep 0 and 1 as converged
        // 2 varies every other.
        genotype.genotype[2] = i % 2;
        // 3 is correlated (opposite)
        genotype.genotype[3] = 1 - i % 2;
        // 4 is 0, 0, 1, 1 repeating.
        genotype.genotype[4] = (i % 4) / 2;
        // 5 is equal to 4, unless 2 is equal to 0, then 5 is `1`.
        genotype.genotype[5] = genotype.genotype[2] == 0 ? 1 : genotype.genotype[4];
        // 6 only has a single 1 at the start.
        genotype.genotype[6] = i == 0 ? 1 : 0;
        // 7 only has a single 0 at the start.
        genotype.genotype[7] = i == 0 ? 0 : 1;

        //       Individual
        //     0 1 2 3 4 5 6 7
        // v_0 0 0 0 0 0 0 0 0
        // v_1 0 0 0 0 0 0 0 0
        // v_2 0 1 0 1 0 1 0 1
        // v_3 1 0 1 0 1 0 1 0
        // v_4 0 0 1 1 0 0 1 1
        // v_5 1 0 1 1 1 0 1 1
        // v_6 1 0 0 0 0 0 0 0
        // v_7 0 1 1 1 1 1 1 1
    }

    //
    SECTION("Converged variables should provide MI of 0")
    {
        // initial population consists of all zeroes
        double c_mi = nmi.compute_linkage(0, 1, subpopulation);
        REQUIRE(c_mi == Approx(0.0));
    }
    SECTION("One converged variable should provide MI of 0")
    {
        // initial population consists of all zeroes
        double c_mi = nmi.compute_linkage(0, 2, subpopulation);
        REQUIRE(c_mi == Approx(0.0));
    }
    SECTION("Correlated (in equal counts) variables should provide MI of 1")
    {
        double c_mi = nmi.compute_linkage(2, 3, subpopulation);
        REQUIRE(c_mi == Approx(1.0));
    }
    SECTION("Uncorrelated variables should provide MI of 0")
    {
        double c_mi = nmi.compute_linkage(2, 4, subpopulation);
        REQUIRE(c_mi == Approx(0.0));
    }
    SECTION("Variables that are somewhat related should provide MI between 0 and 1")
    {
        double c_mi = nmi.compute_linkage(2, 5, subpopulation);
        REQUIRE((c_mi > 0.0 && c_mi < 1.0));
    }
    SECTION("Correlated (in non-equal count) variables should provide MI of 1")
    {
        // Only this case is different from MI due to normalization!
        double c_mi = nmi.compute_linkage(6, 7, subpopulation);
        REQUIRE(c_mi == Approx(1.0));
    }
}

TEST_CASE("Random Linkage", "[Linkage][Analysis]")
{
    std::shared_ptr<Population> pop = std::make_shared<Population>();

    // General problem setup.
    size_t l = 8;
    std::vector<char> alphabet_size(l);
    std::fill(alphabet_size.begin(), alphabet_size.end(), 2);

    pop->registerGlobalData(GenotypeCategoricalData{l, alphabet_size});

    RandomLinkage rndl;
    rndl.setPopulation(pop);
    rndl.registerData();
    pop->registerData<GenotypeCategorical>();
    SECTION("should require an Rng to be present")
    {
        REQUIRE_THROWS(rndl.afterRegisterData());
    }
    Rng rng(42);
    pop->registerGlobalData(rng);

    std::vector<Individual> empty;

    SECTION("values should be between 0 and 1")
    {
        double l = rndl.compute_linkage(0, 1, empty);
        REQUIRE((l >= 0.0 && l <= 1.0));
    }

    SECTION("values should be random (i.e. change, without any underlying change to the population)")
    {
        double l0 = rndl.compute_linkage(0, 1, empty);
        double l1 = rndl.compute_linkage(0, 1, empty);
        REQUIRE(l0 != l1);
    }

    SECTION("should not define a minimum threshold")
    {
        REQUIRE(!rndl.filter_minimum_threshold().has_value());
    }

    SECTION("should not define a  maximum threshold")
    {
        REQUIRE(!rndl.filter_maximum_threshold().has_value());
    }
}

TEST_CASE("Fixed Linkage", "[Linkage][Analysis]")
{
    std::shared_ptr<Population> pop = std::make_shared<Population>();

    size_t l = 5;
    SymMatrix<double> linkage(0, 5);
    size_t k = 0;
    for (size_t i = 0; i < l; ++i)
    {
        for (size_t j = i + 1; j < l; ++j)
        {
            linkage[{i, j}] = k++;
        }
    }

    SECTION("should return the values in the matrix")
    {
        FixedLinkage fl(linkage);
        fl.setPopulation(pop);
        fl.registerData();
        fl.afterRegisterData();

        std::vector<Individual> none;

        for (size_t i = 0; i < l; ++i)
        {
            for (size_t j = i + 1; j < l; ++j)
            {
                REQUIRE(fl.compute_linkage(i, j, none) == linkage[{i, j}]);
            }
        }
    }
}

TEST_CASE("Categorical Univariate FOS", "[Linkage][FoS]")
{
    std::shared_ptr<Population> pop = std::make_shared<Population>();

    // General problem setup.
    size_t l = 8;
    std::vector<char> alphabet_size(l);
    std::fill(alphabet_size.begin(), alphabet_size.end(), 2);

    pop->registerGlobalData(GenotypeCategoricalData{l, alphabet_size});

    CategoricalUnivariateFoS fos;
    fos.setPopulation(pop);
    fos.registerData();
    fos.afterRegisterData();

    // Create reference population
    std::vector<Individual> empty;

    fos.learnFoS(empty);

    SECTION("should return a collection of subsets, each on their own.")
    {
        auto fos_elements = fos.getFoS();
        std::vector<std::vector<size_t>> expected(l);
        for (size_t i = 0; i < l; ++i)
        {
            expected[i].resize(1);
            expected[i][0] = i;
        }

        REQUIRE_THAT(fos_elements, Catch::Matchers::UnorderedEquals(expected));
    }
}

// Helpers for use in tests.
bool operator==(const TreeNode &lhs, const TreeNode &rhs)
{
    if (lhs.left != rhs.left)
        return false;
    if (lhs.right != rhs.right)
        return false;
    if (lhs.distance != rhs.distance)
        return false;
    if (lhs.size != rhs.size)
        return false;
    return true;
}
bool operator!=(const TreeNode &lhs, const TreeNode &rhs)
{
    return !(lhs == rhs);
}
std::ostream &operator<<(std::ostream &lhs, const TreeNode &rhs)
{
    lhs << "TreeNode {\n"
        << "\tleft:\t" << rhs.left << "\n"
        << "\tright:\t" << rhs.right << "\n"
        << "\tdistance:\t" << rhs.distance << "\n"
        << "\tsize:\t" << rhs.size << "\n"
        << "}";
    return lhs;
}

// undo unique naming as this may cause two equivalent trees to have differences.
std::vector<TreeNode> relabel(std::vector<TreeNode> r)
{
    size_t n = r.size() + 1;
    auto get_label = [&r, n](size_t idx) {
        if (idx < n)
            return idx;
        // assume to have been relabeled already
        return std::min(r[idx - n].left, r[idx - n].right);
    };

    for (auto &m : r)
    {
        m.left = get_label(m.left);
        m.right = get_label(m.right);
    }

    return r;
}
TEST_CASE("tree relabel")
{
    std::vector<TreeNode> input = {TreeNode{0, 1, 1.0, 2}, TreeNode{3, 2, 0.0, 3}};

    std::vector<TreeNode> expected = {TreeNode{0, 1, 1.0, 2}, TreeNode{0, 2, 0.0, 3}};

    std::vector<TreeNode> relabeled = relabel(input);

    REQUIRE_THAT(relabeled, Catch::Matchers::Equals(expected));
}

TEST_CASE("performHierarchicalClustering")
{
    SECTION("on a 2x2 matrix")
    {
        size_t seed = GENERATE(42, 43, 44);
        Rng rng(seed);
        SymMatrix<double> in(0, 2);
        in.set(0, 1, 1.0);

        std::vector<TreeNode> expected = {TreeNode{0, 1, 1.0, 2}};

        auto tree = performHierarchicalClustering(in, rng);
        REQUIRE_THAT(tree, Catch::Matchers::UnorderedEquals(expected));
    }

    SECTION("on a 3x3 matrix")
    {
        size_t seed = GENERATE(42, 43, 44);
        Rng rng(seed);
        SymMatrix<double> in(0, 3);
        in.set(0, 1, 1.0);

        // First merge sets containing 0 and 1 (indices: 0 and 1)
        // Then merge sets containing (0, 1) and (2) (indices 3 and 2)
        std::vector<TreeNode> expected = {TreeNode{0, 1, 1.0, 2}, TreeNode{3, 2, 0.0, 3}};

        auto tree = performHierarchicalClustering(in, rng);
        REQUIRE_THAT(tree, Catch::Matchers::UnorderedEquals(expected));
    }

    SECTION("on a 3x3 zero matrix - should be random")
    {
        size_t seed = GENERATE(42, 43, 44);
        Rng rng(seed);
        SymMatrix<double> in(0, 3);

        /**
         * (1/3)^num_trails chance of failing this test
         * as there are three possible trees (ignoring order):
         *     R        R       R
         *    / \      / \     / \
         *   3   2    3   1   0   3
         *  / \      / \          / \
         * 0   1    0   2        1   2
         **/
        size_t num_trails = 20;

        auto first_tree = performHierarchicalClustering(in, rng);
        bool any_nonequal = false;
        for (size_t trail = 0; trail < num_trails; ++trail)
        {
            auto second_tree = performHierarchicalClustering(in, rng);
            // The number of merges is fixed.
            REQUIRE(first_tree.size() == second_tree.size());
            // But the merges themselves may differ.
            if (!std::is_permutation(first_tree.begin(), first_tree.end(), second_tree.begin()))
            {
                // std::cout << "Number of trials required: " << trail + 1 << "\n";
                any_nonequal = true;
                break;
            }
        }
        REQUIRE(any_nonequal);
    }

    SECTION("on 4x4 matrix: symmetric tree")
    {
        size_t seed = GENERATE(42, 43, 44);
        Rng rng(seed);
        SymMatrix<double> in(0, 4);

        in.set(0, 1, 1);
        in.set(2, 3, 1);

        // First - merge sets containing 0 and 1 (indices: 0 and 1)
        //       - merge sets containing 2 and 3 (indices: 0 and 1)
        //  - in any order.
        // Then merge sets containing (0, 1) and (2, 3) (indices 3 and 4)
        std::vector<TreeNode> expected = {TreeNode{0, 1, 1.0, 2}, TreeNode{2, 3, 1.0, 2}, TreeNode{0, 2, 0.0, 4}};

        auto tree = relabel(performHierarchicalClustering(in, rng));
        REQUIRE_THAT(tree, Catch::Matchers::UnorderedEquals(expected));
    }

    SECTION("on a 4x4 matrix: list-like tree")
    {
        // Use many seeds, we want to cover the case where it starts at 4,
        // then goes to 3, then 2, then 1.
        // as this happens in one of 4! = 4*3*2=24 permutations
        // we need plenty of seeds to make this likely to happen.
        size_t seed = GENERATE(range(42, 242));
        Rng rng(seed);
        SymMatrix<double> in(0, 4);

        in.set(0, 1, 1);
        in.set(1, 2, 0.5);
        in.set(0, 2, 0.5);

        // First merge sets containing 0 and 1 (indices: 0 and 1)
        // Then merge sets containing (0, 1) and (2) (indices 4 and 2)
        // Then merge sets containing (0, 1, 2) and (3) (indices 5 and 3)
        std::vector<TreeNode> expected = {TreeNode{0, 1, 1.0, 2}, TreeNode{0, 2, 0.5, 3}, TreeNode{0, 3, 0.0, 4}};

        auto tree = relabel(performHierarchicalClustering(in, rng));
        REQUIRE_THAT(tree, Catch::Matchers::UnorderedEquals(expected));
    }

    SECTION("on a 5x5 matrix")
    {
        size_t seed = GENERATE(42, 43, 44);
        Rng rng(seed);
        SymMatrix<double> in(0, 5);
        in.set(0, 1, 1);
        in.set(2, 3, 1);
        in.set(0, 2, 0.5);
        in.set(0, 3, 0.5);
        in.set(1, 2, 0.5);
        in.set(1, 3, 0.5);

        std::vector<TreeNode> expected = {
            TreeNode{0, 1, 1.0, 2}, TreeNode{2, 3, 1.0, 2}, TreeNode{0, 2, 0.5, 4}, TreeNode{0, 4, 0.0, 5}};

        auto tree = relabel(performHierarchicalClustering(in, rng));
        REQUIRE_THAT(tree, Catch::Matchers::UnorderedEquals(expected));
    }
}

bool verify_order_nop(FoS &)
{
    return true;
}

bool verify_order_sorted_size(FoS &fos)
{
    size_t largest_size_so_far = 0;
    for (auto &e : fos)
    {
        largest_size_so_far = std::max(largest_size_so_far, e.size());
        if (e.size() < largest_size_so_far)
        {
            return false;
        }
    }
    return true;
}

struct CLTConfiguration
{
    size_t l;
    FoSOrdering ordering;
    bool filter_root;
    std::optional<double> filter_minimum_threshold;
    bool filter_minima;
    std::optional<double> filter_maximum_threshold;
    bool filter_maxima;
    SymMatrix<double> similarity;
    FoS expected;
    std::function<bool(FoS &)> verify_order;
};

TEST_CASE("Categorical Linkage Tree")
{
    SECTION("deterministic cases")
    {
        CLTConfiguration configuration =
            GENERATE(CLTConfiguration{4,
                                      AsIs,
                                      false, // Do not remove the root
                                      std::nullopt,
                                      false, // Do not remove elements below the minimum threshold
                                      std::nullopt,
                                      false, // Do not remove children of elements above the maximum threshold
                                      SymMatrix<double>({{0.0, 1.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 1.0}, {0.0}}),
                                      {{0}, {1}, {2}, {3}, {0, 1}, {2, 3}, {0, 1, 2, 3}},
                                      verify_order_nop},
                     CLTConfiguration{4,
                                      AsIs,
                                      true, // Do remove the root
                                      std::nullopt,
                                      false, // Do not remove elements below the minimum threshold
                                      std::nullopt,
                                      false, // Do not remove children of elements above the maximum threshold
                                      SymMatrix<double>({{0.0, 1.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 1.0}, {0.0}}),
                                      {{0}, {1}, {2}, {3}, {0, 1}, {2, 3}},
                                      verify_order_nop},
                     CLTConfiguration{4,
                                      SizeIncreasing,
                                      true, // Do remove the root
                                      std::nullopt,
                                      false, // Do not remove elements below the minimum threshold
                                      std::nullopt,
                                      false, // Do not remove children of elements above the maximum threshold
                                      SymMatrix<double>({{0.0, 1.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 1.0}, {0.0}}),
                                      {{0}, {1}, {2}, {3}, {0, 1}, {2, 3}},
                                      verify_order_sorted_size},
                     CLTConfiguration{4,
                                      AsIs,
                                      false, // Do not remove the root
                                      {0.0},
                                      true, // Do remove elements below the minimum threshold
                                      std::nullopt,
                                      false, // Do not remove children of elements above the maximum threshold
                                      SymMatrix<double>({{0.0, 1.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 1.0}, {0.0}}),
                                      {{0}, {1}, {2}, {3}, {0, 1}, {2, 3}},
                                      verify_order_nop},
                     CLTConfiguration{4,
                                      AsIs,
                                      true, // Do remove the root
                                      std::nullopt,
                                      false, // Do not remove elements below the minimum threshold
                                      {1.0},
                                      true, // Do remove children of elements above the maximum threshold
                                      SymMatrix<double>({{0.0, 1.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 1.0}, {0.0}}),
                                      {{0, 1}, {2, 3}},
                                      verify_order_nop});
        auto l = configuration.l;
        auto ordering = configuration.ordering;
        auto filter_root = configuration.filter_root;
        auto minimum_th = configuration.filter_minimum_threshold;
        auto filter_minima = configuration.filter_minima;
        auto maximum_th = configuration.filter_maximum_threshold;
        auto filter_maxima = configuration.filter_maxima;
        auto similarity = configuration.similarity;
        auto expected = configuration.expected;
        auto verify_order = configuration.verify_order;
        // std::cout << "Configuration: {"
        //     << "\tl : " << l << '\n'
        //     << "\tordering : " << ordering << '\n'
        //     << "\tfilter_root : " << filter_root << '\n'
        //     << "\tminimum threshold : " << (minimum_th.has_value() ? std::to_string(minimum_th.value()) : "[none]" )
        //     << '\n'
        //     << "\tfilter_minima : " << filter_minima << '\n'
        //     << "\tmaximum threshold : " << (maximum_th.has_value() ? std::to_string(maximum_th.value()) : "[none]" )
        //     << '\n'
        //     << "\tfilter_maxima : " << filter_maxima << '\n'
        //     // skipping similarity matrix & expected & verify_order
        //     << "}\n";

        std::shared_ptr<Population> pop = std::make_shared<Population>();
        Rng rng(42);

        std::vector<char> alphabet_size(l);
        std::fill(alphabet_size.begin(), alphabet_size.end(), 2);

        pop->registerGlobalData(GenotypeCategoricalData {l, alphabet_size});

        std::shared_ptr<LinkageMetric> metric(new FixedLinkage(similarity, minimum_th, maximum_th));
        CategoricalLinkageTree lt(metric, ordering, filter_minima, filter_maxima, filter_root);
        lt.setPopulation(pop);
        lt.registerData();
        lt.afterRegisterData();

        std::vector<Individual> empty;
        SECTION("requires the Rng to be present")
        {
            REQUIRE_THROWS(lt.learnFoS(empty));
        }
        pop->registerGlobalData(rng);

        SECTION("returns the right subsets")
        {
            lt.learnFoS(empty);

            FoS &fos = lt.getFoS();

            REQUIRE_THAT(fos, Catch::Matchers::UnorderedEquals(expected));
            REQUIRE(verify_order(fos));
        }
    }

    SECTION("random cases")
    {
        std::shared_ptr<Population> pop = std::make_shared<Population>();
        Rng rng(42);

        size_t l = 4;
        SymMatrix<double> similarity = SymMatrix<double>({{0.0, 1.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 1.0}, {0.0}});

        std::vector<char> alphabet_size(l);
        std::fill(alphabet_size.begin(), alphabet_size.end(), 2);

        pop->registerGlobalData(GenotypeCategoricalData {l, alphabet_size});

        std::shared_ptr<LinkageMetric> metric(new FixedLinkage(similarity));
        CategoricalLinkageTree lt(metric, Random, false, false, false);
        lt.setPopulation(pop);
        lt.registerData();
        lt.afterRegisterData();
        std::vector<Individual> empty;

        SECTION("requires the Rng to be present")
        {
            REQUIRE_THROWS(lt.learnFoS(empty));
        }
        pop->registerGlobalData(rng);

        SECTION("output should ordered randomly, i.e. nondeterministically")
        {
            // FoS has 7 elements, and as such 7! possible orderings: 1/5.040 chance of failure.
            // a maximum 3 trails nets us a failure probability < 1e-10 if everything is in order.
            size_t max_trails = 3;
            bool success = false;
            for (size_t trail = 0; trail < max_trails; ++trail)
            {
                lt.learnFoS(empty);
                // this is necessarily a copy, otherwise fos_a and fos_b will always refer
                // to the same memory, which is always equal.
                FoS fos_a = lt.getFoS();
                lt.learnFoS(empty);
                FoS fos_b = lt.getFoS();
                if (!std::equal(fos_a.begin(), fos_a.end(), fos_b.begin()))
                {
                    success = true;
                    break;
                }
            }
            REQUIRE(success);
        }
    }
}

TEST_CASE("CategoricalPopulationSamplingDistribution")
{
    std::shared_ptr<Population> pop = std::make_shared<Population>();
    Rng rng(42);

    // size_t l = 4;
    size_t subset_id = GENERATE(range(0b0000, 0b1111));

    pop->registerData<GenotypeCategorical>();
    pop->registerGlobalData(rng);

    Individual current = pop->newIndividual();

    std::vector<Individual> subpopulation(1);
    Individual donor = pop->newIndividual();
    subpopulation[0] = donor;

    GenotypeCategorical &current_genotype = pop->getData<GenotypeCategorical>(current);
    current_genotype.genotype.resize(4);
    std::fill(current_genotype.genotype.begin(), current_genotype.genotype.end(), 0);

    GenotypeCategorical &donor_genotype = pop->getData<GenotypeCategorical>(donor);
    donor_genotype.genotype.resize(4);
    std::fill(donor_genotype.genotype.begin(), donor_genotype.genotype.end(), 1);

    CategoricalPopulationSamplingDistribution dist(*pop, subpopulation);

    // Decode subset id
    std::vector<size_t> subset;
    std::vector<size_t> inv_subset;
    if ((subset_id & 1) > 0)
        subset.push_back(0);
    else
        inv_subset.push_back(0);
    if ((subset_id & 2) > 0)
        subset.push_back(1);
    else
        inv_subset.push_back(1);
    if ((subset_id & 4) > 0)
        subset.push_back(2);
    else
        inv_subset.push_back(2);
    if ((subset_id & 8) > 0)
        subset.push_back(3);
    else
        inv_subset.push_back(3);

    // apply change
    dist.apply_resample(current, subset);

    // assert that the change was applied correctly
    for (size_t i : subset)
        CHECK(current_genotype.genotype[i] == 1);

    for (size_t i : inv_subset)
        CHECK(current_genotype.genotype[i] == 0);
}

TEST_CASE("CategoricalDonorSearchDistribution")
{
    std::shared_ptr<Population> pop = std::make_shared<Population>();
    Rng rng(42);

    // size_t l = 4;
    size_t subset_id = GENERATE(range(0b0000, 0b1111));

    pop->registerData<GenotypeCategorical>();
    pop->registerGlobalData(rng);

    Individual current = pop->newIndividual();

    std::vector<Individual> subpopulation(2);
    Individual donor = pop->newIndividual();
    Individual duplicate = pop->newIndividual();
    subpopulation[0] = donor;
    // Having a duplicate solution should not impact the test in case of donor search.
    subpopulation[1] = duplicate;

    GenotypeCategorical &current_genotype = pop->getData<GenotypeCategorical>(current);
    current_genotype.genotype.resize(4);
    std::fill(current_genotype.genotype.begin(), current_genotype.genotype.end(), 0);

    GenotypeCategorical &donor_genotype = pop->getData<GenotypeCategorical>(donor);
    donor_genotype.genotype.resize(4);
    std::fill(donor_genotype.genotype.begin(), donor_genotype.genotype.end(), 1);

    pop->copyIndividual(current, duplicate);

    CategoricalDonorSearchDistribution dist(*pop, subpopulation);

    // Decode subset id
    std::vector<size_t> subset;
    std::vector<size_t> inv_subset;
    if ((subset_id & 1) > 0)
        subset.push_back(0);
    else
        inv_subset.push_back(0);
    if ((subset_id & 2) > 0)
        subset.push_back(1);
    else
        inv_subset.push_back(1);
    if ((subset_id & 4) > 0)
        subset.push_back(2);
    else
        inv_subset.push_back(2);
    if ((subset_id & 8) > 0)
        subset.push_back(3);
    else
        inv_subset.push_back(3);

    // apply change
    dist.apply_resample(current, subset);

    // assert that the change was applied correctly
    for (size_t i : subset)
        CHECK(current_genotype.genotype[i] == 1);

    for (size_t i : inv_subset)
        CHECK(current_genotype.genotype[i] == 0);
}

TEST_CASE("GOM")
{
    using trompeloeil::_;

    std::shared_ptr<Population> pop = std::make_shared<Population>();

    std::shared_ptr<MockSamplingDistribution> distribution(new MockSamplingDistribution());
    std::shared_ptr<MockObjectiveFunction> of(new MockObjectiveFunction());
    std::shared_ptr<MockAcceptanceCriterion> acceptance_criterion(new MockAcceptanceCriterion());
    pop->registerGlobalData(GObjectiveFunction(of.get()));

    ALLOW_CALL(*of, setPopulation(_));
    ALLOW_CALL(*of, registerData());
    ALLOW_CALL(*of, afterRegisterData());
    ALLOW_CALL(*acceptance_criterion, setPopulation(_));
    ALLOW_CALL(*acceptance_criterion, registerData());
    ALLOW_CALL(*acceptance_criterion, afterRegisterData());

    GOM gom;
    gom.setPopulation(pop);

    Individual ii = pop->newIndividual();

    std::vector<std::vector<size_t>> fos = {{0}, {1}, {2}};

    trompeloeil::sequence seq0, seq1, seq2;

    // Change -> Evaluate -> Judge
    REQUIRE_CALL(*distribution, apply_resample(ii, fos[0])).RETURN(true).IN_SEQUENCE(seq0);
    REQUIRE_CALL(*of, evaluate(ii)).IN_SEQUENCE(seq0);
    REQUIRE_CALL(*acceptance_criterion, compare(_, ii)).RETURN(1).IN_SEQUENCE(seq0);

    REQUIRE_CALL(*distribution, apply_resample(ii, fos[1])).RETURN(true).IN_SEQUENCE(seq1);
    REQUIRE_CALL(*of, evaluate(ii)).IN_SEQUENCE(seq1);
    REQUIRE_CALL(*acceptance_criterion, compare(_, ii)).RETURN(2).IN_SEQUENCE(seq1);

    REQUIRE_CALL(*distribution, apply_resample(ii, fos[2])).RETURN(true).IN_SEQUENCE(seq2);
    REQUIRE_CALL(*of, evaluate(ii)).IN_SEQUENCE(seq2);
    REQUIRE_CALL(*acceptance_criterion, compare(_, ii)).RETURN(3).IN_SEQUENCE(seq2);

    bool changed = false;
    bool improved = false;
    gom.apply(ii, fos, distribution.get(), acceptance_criterion.get(), changed, improved);
}

TEST_CASE("FI")
{
    using trompeloeil::_;

    std::shared_ptr<Population> pop = std::make_shared<Population>();

    std::shared_ptr<MockSamplingDistribution> distribution(new MockSamplingDistribution());
    std::shared_ptr<MockObjectiveFunction> of(new MockObjectiveFunction());
    std::shared_ptr<MockAcceptanceCriterion> acceptance_criterion(new MockAcceptanceCriterion());
    pop->registerGlobalData(GObjectiveFunction(of.get()));

    ALLOW_CALL(*of, setPopulation(_));
    ALLOW_CALL(*of, registerData());
    ALLOW_CALL(*of, afterRegisterData());
    ALLOW_CALL(*acceptance_criterion, setPopulation(_));
    ALLOW_CALL(*acceptance_criterion, registerData());
    ALLOW_CALL(*acceptance_criterion, afterRegisterData());

    FI fi;
    fi.setPopulation(pop);

    Individual ii = pop->newIndividual();

    std::vector<std::vector<size_t>> fos = {{0}, {1}, {2}, {3}};

    trompeloeil::sequence seq0, seq1, seq2;

    // TODO: Check if evlauation and accepting is skipped if sampling did not update.
    // Change -> Evaluate -> Judge for 0 - 2
    REQUIRE_CALL(*distribution, apply_resample(ii, fos[0])).RETURN(true).IN_SEQUENCE(seq0);
    REQUIRE_CALL(*of, evaluate(ii)).IN_SEQUENCE(seq0);
    REQUIRE_CALL(*acceptance_criterion, compare(_, ii)).RETURN(3).IN_SEQUENCE(seq0);

    REQUIRE_CALL(*distribution, apply_resample(ii, fos[1])).RETURN(true).IN_SEQUENCE(seq1);
    REQUIRE_CALL(*of, evaluate(ii)).IN_SEQUENCE(seq1);
    REQUIRE_CALL(*acceptance_criterion, compare(_, ii)).RETURN(1).IN_SEQUENCE(seq1);

    REQUIRE_CALL(*distribution, apply_resample(ii, fos[2])).RETURN(true).IN_SEQUENCE(seq2);
    REQUIRE_CALL(*of, evaluate(ii)).IN_SEQUENCE(seq2);
    REQUIRE_CALL(*acceptance_criterion, compare(_, ii)).RETURN(2).IN_SEQUENCE(seq2);

    // As previous was successful, abort.
    FORBID_CALL(*distribution, apply_resample(ii, fos[3]));

    bool changed = false;
    bool improved = false;
    fi.apply(ii, fos, distribution.get(), acceptance_criterion.get(), changed, improved);
}

TEST_CASE("Integration Test: GOMEA on OneMax")
{
    std::shared_ptr<Population> pop = std::make_shared<Population>();

    size_t l = 100;
    size_t population_size = 16;
    std::shared_ptr<ISolutionInitializer> initializer(new CategoricalProbabilisticallyCompleteInitializer());
    std::shared_ptr<LinkageMetric> metric(new NMI());
    std::shared_ptr<FoSLearner> foslearner(new CategoricalLinkageTree(metric));
    std::shared_ptr<IPerformanceCriterion> performance_criterion(new SingleObjectiveAcceptanceCriterion());
    std::shared_ptr<IArchive> archive(new BruteforceArchive({0}));

    // std::shared_ptr<GOM> gom(new GOM(performance_criterion));
    // std::shared_ptr<FI> fi(new FI(performance_criterion));
    //
    Rng rng(42);
    pop->registerGlobalData(rng);
    OneMax onemax(l);

    onemax.setPopulation(pop);
    pop->registerGlobalData(GObjectiveFunction(&onemax));

    GOMEA gomea(population_size, initializer, foslearner, performance_criterion, archive);
    gomea.setPopulation(pop);
    onemax.registerData();
    gomea.registerData();
    onemax.afterRegisterData();
    gomea.afterRegisterData();

    gomea.step();
    gomea.step();
    gomea.step();
    gomea.step();
    gomea.step();

    Objective &objective = pop->getData<Objective>(archive->get_archived()[0]);
    REQUIRE(objective.objectives.size() == 1);
    REQUIRE(objective.objectives[0] == -100);
}

TEST_CASE("Integration Test: GOMEA on Deceptive Trap Function")
{
    std::shared_ptr<Population> pop = std::make_shared<Population>();

    size_t l = 100;
    size_t population_size = 128;
    std::shared_ptr<ISolutionInitializer> initializer(new CategoricalProbabilisticallyCompleteInitializer());
    std::shared_ptr<LinkageMetric> metric(new NMI());
    std::shared_ptr<FoSLearner> foslearner(new CategoricalLinkageTree(metric));
    std::shared_ptr<IPerformanceCriterion> performance_criterion(new SingleObjectiveAcceptanceCriterion());
    // std::shared_ptr<GOM> gom(new GOM(performance_criterion));
    // std::shared_ptr<FI> fi(new FI(performance_criterion));
    //
    Rng rng(42);
    pop->registerGlobalData(rng);
    std::shared_ptr<IArchive> archive(new BruteforceArchive({0}));

    std::vector<size_t> permutation_in_order(l);
    std::iota(permutation_in_order.begin(), permutation_in_order.end(), 0);
    std::vector<char> optimum(l);
    std::fill(optimum.begin(), optimum.end(), 1);
    ConcatenatedPermutedTrap cpt{l, 5, permutation_in_order, optimum};
    BestOfTraps bot(BestOfTrapsInstance{l, {cpt}});
    bot.setPopulation(pop);
    pop->registerGlobalData(GObjectiveFunction(&bot));

    GOMEA gomea(population_size, initializer, foslearner, performance_criterion, archive);
    gomea.setPopulation(pop);
    bot.registerData();
    gomea.registerData();
    bot.afterRegisterData();
    gomea.afterRegisterData();

    gomea.step();
    gomea.step();
    gomea.step();
    gomea.step();
    gomea.step();
    gomea.step();
    gomea.step();
    gomea.step();
    gomea.step();

    Objective &objective = pop->getData<Objective>(archive->get_archived()[0]);
    REQUIRE(objective.objectives.size() == 1);
    REQUIRE(objective.objectives[0] == -100);
}

TEST_CASE("compute_objective_ranges")
{
    std::shared_ptr<Population> pop = std::make_shared<Population>();
    pop->registerData<Objective>();

    std::vector<size_t> objective_indices = {0, 2};
    std::vector<Individual> individuals(3);
    std::generate(individuals.begin(), individuals.end(), [&pop] { return pop->newIndividual(); });

    auto &o0 = pop->getData<Objective>(individuals[0]);
    o0.objectives = {0.0, 0.5, 1.0};
    auto &o1 = pop->getData<Objective>(individuals[1]);
    o1.objectives = {1.5, 0.5, 0.5};
    auto &o2 = pop->getData<Objective>(individuals[2]);
    o2.objectives = {1.0, 0.5, 0.0};

    auto res = compute_objective_ranges(*pop, objective_indices, individuals);
    REQUIRE(res.size() == 2);
    REQUIRE(res[0] == 1.5);
    REQUIRE(res[1] == 1.0);
}

TEST_CASE("determine_extreme_clusters")
{
    std::vector<ObjectiveCluster> clusters = {
        ObjectiveCluster{{1.0, 0.0}, {}},
        ObjectiveCluster{{0.5, 0.5}, {}},
        ObjectiveCluster{{0.0, 1.0}, {}},
    };
    std::vector<size_t> objective_indices {0, 1};
    determine_extreme_clusters(objective_indices, clusters);
    REQUIRE(clusters[0].mixing_mode == 1);
    REQUIRE(clusters[1].mixing_mode == -1);
    REQUIRE(clusters[2].mixing_mode == 0);
}

TEST_CASE("cluster_mo_gomea")
{

    std::shared_ptr<Population> pop = std::make_shared<Population>();
    pop->registerData<Objective>();
    Rng rng(42);
    pop->registerGlobalData(rng);

    std::vector<size_t> objective_indices = {0, 2};
    std::vector<Individual> individuals(5);
    std::generate(individuals.begin(), individuals.end(), [&pop] { return pop->newIndividual(); });

    auto &o0 = pop->getData<Objective>(individuals[0]);
    o0.objectives = {0.0, 0.5, 1.0};
    auto &o1 = pop->getData<Objective>(individuals[1]);
    o1.objectives = {0.0, 0.5, 1.0};
    auto &o2 = pop->getData<Objective>(individuals[2]);
    o2.objectives = {1.5, 0.5, 0.5};
    auto &o3 = pop->getData<Objective>(individuals[3]);
    o3.objectives = {1.0, 0.5, 0.0};
    auto &o4 = pop->getData<Objective>(individuals[4]);
    o4.objectives = {1.0, 0.5, 0.0};

    auto ranges = compute_objective_ranges(*pop, objective_indices, individuals);

    SECTION("1 cluster")
    {
        auto res = cluster_mo_gomea(*pop, individuals, objective_indices, ranges, 1);
        REQUIRE(res.size() == 1);
        REQUIRE_THAT(res[0].members, Catch::Matchers::UnorderedEquals(individuals));
    }

    SECTION("2 clusters")
    {
        auto res = cluster_mo_gomea(*pop, individuals, objective_indices, ranges, 2);
        REQUIRE(res.size() == 2);
        for (auto &cluster : res)
        {
            size_t expected_size = (individuals.size() * 2) / 2;
            REQUIRE(cluster.members.size() == expected_size);
        }
    }

    SECTION("3 clusters")
    {
        auto res = cluster_mo_gomea(*pop, individuals, objective_indices, ranges, 3);
        REQUIRE(res.size() == 3);
        for (auto &cluster : res)
        {
            size_t expected_size = (individuals.size() * 2) / 3;
            REQUIRE(cluster.members.size() == expected_size);
        }
    }

    SECTION("4 clusters")
    {
        auto res = cluster_mo_gomea(*pop, individuals, objective_indices, ranges, 4);
        REQUIRE(res.size() == 4);
        for (auto &cluster : res)
        {
            size_t expected_size = (individuals.size() * 2) / 4;
            REQUIRE(cluster.members.size() == expected_size);
        }
    }

    SECTION("5 clusters")
    {
        auto res = cluster_mo_gomea(*pop, individuals, objective_indices, ranges, 5);
        REQUIRE(res.size() == 5);
        for (auto &cluster : res)
        {
            size_t expected_size = (individuals.size() * 2) / 5;
            REQUIRE(cluster.members.size() == expected_size);
        }
    }
}

TEST_CASE("determine_cluster_to_use_mo_gomea")
{
    std::shared_ptr<Population> pop = std::make_shared<Population>();
    pop->registerData<Objective>();
    pop->registerData<ClusterIndex>();
    Rng rng(42);
    pop->registerGlobalData(rng);

    std::vector<size_t> objective_indices = {0, 2};
    std::vector<Individual> individuals(5);
    std::generate(individuals.begin(), individuals.end(), [&pop] { return pop->newIndividual(); });

    auto &o0 = pop->getData<Objective>(individuals[0]);
    o0.objectives = {0.0, 0.5, 1.0};
    auto &o1 = pop->getData<Objective>(individuals[1]);
    o1.objectives = {0.0, 0.5, 1.0};
    auto &o2 = pop->getData<Objective>(individuals[2]);
    o2.objectives = {0.5, 0.5, 1.0};
    auto &o3 = pop->getData<Objective>(individuals[3]);
    o3.objectives = {1.0, 0.5, 0.0};
    auto &o4 = pop->getData<Objective>(individuals[4]);
    o4.objectives = {1.0, 0.5, 0.0};

    auto ranges = compute_objective_ranges(*pop, objective_indices, individuals);

    std::vector<ObjectiveCluster> clusters = {{{0.0, 1.0}, {individuals[0], individuals[1]}},
                                              {{1.0, 0.0}, {individuals[3], individuals[4]}}};

    determine_cluster_to_use_mo_gomea(*pop, clusters, individuals, objective_indices, ranges);

    // Assigned to own cluster
    auto &ci0 = pop->getData<ClusterIndex>(individuals[0]);
    REQUIRE(ci0.cluster_index == 0);
    auto &ci1 = pop->getData<ClusterIndex>(individuals[1]);
    REQUIRE(ci1.cluster_index == 0);
    auto &ci3 = pop->getData<ClusterIndex>(individuals[3]);
    REQUIRE(ci3.cluster_index == 1);
    auto &ci4 = pop->getData<ClusterIndex>(individuals[4]);
    REQUIRE(ci4.cluster_index == 1);

    // Nearest
    auto &ci2 = pop->getData<ClusterIndex>(individuals[2]);
    REQUIRE(ci2.cluster_index == 0);
}

TEST_CASE("hoang_generate_initial_weights")
{
    SECTION("assigns weights to all provided individuals") {
        auto population = std::make_shared<Population>();
        population->registerGlobalData(Rng(42));
        TschebysheffObjectiveScalarizer scalarizer({0, 1});
        scalarizer.scalarizer_id = 0;
        scalarizer.setPopulation(population);
        scalarizer.registerData();
        population->registerData<Objective>();
        population->registerData<ScalarizationWeights>();
        scalarizer.afterRegisterData();

        auto obj = population->getDataContainer<Objective>();
        auto sw = population->getDataContainer<ScalarizationWeights>();

        std::vector<Individual> individuals(2);
        population->newIndividuals(individuals);

        auto &obj_ind0 = obj.getData(individuals[0]);
        obj_ind0.objectives = {0.0, 1.0};
        auto &obj_ind1 = obj.getData(individuals[1]);
        obj_ind1.objectives = {1.0, 0.0};

        hoang_generate_initial_weights(
            *population,
            scalarizer,
            scalarizer.get_dim(),
            individuals);

        auto &weights_ind0 =sw.getData(individuals[0]);
        auto &weights_ind1 =sw.getData(individuals[1]);
        // We have one scalarizer with index 0 - so size should be 1.
        REQUIRE(weights_ind0.weights.size() == 1);
        REQUIRE(weights_ind1.weights.size() == 1);
        // Two objectives are scalarized by the scalarizer - hence two weights.
        REQUIRE(weights_ind0.weights[0].size() == 2);
        REQUIRE(weights_ind1.weights[0].size() == 2);
        // Solution 0 outperforms in objective 0 (i.e. is the lowest) - so most of the weight should go
        // to this variable to minimize the scalarization.
        REQUIRE(weights_ind0.weights[0][0] > weights_ind1.weights[0][0]);
        // Other way around for solution 1.
        REQUIRE(weights_ind0.weights[0][1] < weights_ind1.weights[0][1]);
        // No vector should be uninitialized
        REQUIRE(is_any_individual_scalarizationweight_uninit(sw, individuals, 0, 2) == 0);
    }

    SECTION("assigns weights to all provided individuals even without Objective values") {
        auto population = std::make_shared<Population>();
        population->registerGlobalData(Rng(42));
        TschebysheffObjectiveScalarizer scalarizer({0, 1});
        scalarizer.scalarizer_id = 0;
        scalarizer.setPopulation(population);
        scalarizer.registerData();
        population->registerData<Objective>();
        population->registerData<ScalarizationWeights>();
        scalarizer.afterRegisterData();

        auto obj = population->getDataContainer<Objective>();
        auto sw = population->getDataContainer<ScalarizationWeights>();

        std::vector<Individual> individuals(10);
        population->newIndividuals(individuals);

        auto &obj_ind2 = obj.getData(individuals[2]);
        obj_ind2.objectives = {1.0, 1.0};

        hoang_generate_initial_weights(
            *population,
            scalarizer,
            scalarizer.get_dim(),
            individuals);

        auto &weights_ind2 =sw.getData(individuals[2]);
        auto &weights_ind6 =sw.getData(individuals[6]);
        // We have one scalarizer with index 0 - so size should be 1.
        REQUIRE(weights_ind2.weights.size() == 1);
        REQUIRE(weights_ind6.weights.size() == 1);
        // Two objectives are scalarized by the scalarizer - hence two weights.
        REQUIRE(weights_ind2.weights[0].size() == 2);
        REQUIRE(weights_ind6.weights[0].size() == 2);
        // No vector should be uninitialized
        REQUIRE(is_any_individual_scalarizationweight_uninit(sw, individuals, 0, 2) == 0);
    }
}

TEST_CASE("hoang_reassign_weights")
{
    SECTION("assigns weights to all provided individuals") {
        auto population = std::make_shared<Population>();
        population->registerGlobalData(Rng(42));
        TschebysheffObjectiveScalarizer scalarizer({0, 1});
        scalarizer.scalarizer_id = 0;
        scalarizer.setPopulation(population);
        scalarizer.registerData();
        population->registerData<Objective>();
        population->registerData<ScalarizationWeights>();
        scalarizer.afterRegisterData();

        auto obj = population->getDataContainer<Objective>();
        auto sw = population->getDataContainer<ScalarizationWeights>();

        std::vector<Individual> individuals(2);
        population->newIndividuals(individuals);

        auto &obj_ind0 = obj.getData(individuals[0]);
        obj_ind0.objectives = {0.0, 1.0};
        auto &obj_ind1 = obj.getData(individuals[1]);
        obj_ind1.objectives = {1.0, 0.0};

        hoang_generate_initial_weights(
            *population,
            scalarizer,
            scalarizer.get_dim(),
            individuals);

        // Swap them around
        obj_ind0.objectives = {1.0, 0.0};
        obj_ind1.objectives = {0.0, 1.0};

        hoang_reassign_weights(
            *population,
            scalarizer,
            scalarizer.get_dim(),
            individuals);

        auto &weights_ind0 =sw.getData(individuals[0]);
        auto &weights_ind1 =sw.getData(individuals[1]);
        // We have one scalarizer with index 0 - so size should be 1.
        REQUIRE(weights_ind0.weights.size() == 1);
        REQUIRE(weights_ind1.weights.size() == 1);
        // Two objectives are scalarized by the scalarizer - hence two weights.
        REQUIRE(weights_ind0.weights[0].size() == 2);
        REQUIRE(weights_ind1.weights[0].size() == 2);
        // The order of solutions should now be flipped around compared to the original
        // setting.
        REQUIRE(weights_ind0.weights[0][0] < weights_ind1.weights[0][0]);
        REQUIRE(weights_ind0.weights[0][1] > weights_ind1.weights[0][1]);
        // No vector should be uninitialized
        REQUIRE(is_any_individual_scalarizationweight_uninit(sw, individuals, 0, 2) == 0);
    }

    SECTION("assigns weights to all provided individuals even without Objective values") {
        auto population = std::make_shared<Population>();
        population->registerGlobalData(Rng(42));
        TschebysheffObjectiveScalarizer scalarizer({0, 1});
        scalarizer.scalarizer_id = 0;
        scalarizer.setPopulation(population);
        scalarizer.registerData();
        population->registerData<Objective>();
        population->registerData<ScalarizationWeights>();
        scalarizer.afterRegisterData();

        auto obj = population->getDataContainer<Objective>();
        auto sw = population->getDataContainer<ScalarizationWeights>();

        std::vector<Individual> individuals(10);
        population->newIndividuals(individuals);

        auto &obj_ind2 = obj.getData(individuals[2]);
        obj_ind2.objectives = {1.0, 1.0};

        hoang_generate_initial_weights(
            *population,
            scalarizer,
            scalarizer.get_dim(),
            individuals);

        hoang_reassign_weights(
            *population,
            scalarizer,
            scalarizer.get_dim(),
            individuals);

        auto &weights_ind2 =sw.getData(individuals[2]);
        auto &weights_ind6 =sw.getData(individuals[6]);
        // We have one scalarizer with index 0 - so size should be 1.
        REQUIRE(weights_ind2.weights.size() == 1);
        REQUIRE(weights_ind6.weights.size() == 1);
        // Two objectives are scalarized by the scalarizer - hence two weights.
        REQUIRE(weights_ind2.weights[0].size() == 2);
        REQUIRE(weights_ind6.weights[0].size() == 2);
        // No vector should be uninitialized
        REQUIRE(is_any_individual_scalarizationweight_uninit(sw, individuals, 0, 2) == 0);
    }
}

TEST_CASE("Integration Test: MO-GOMEA on OneMax vs ZeroMax")
{
    std::shared_ptr<Population> pop = std::make_shared<Population>();

    size_t l = 100;
    size_t population_size = 16;
    size_t number_of_clusters = 3;
    std::vector<size_t> objective_indices = {0, 1};

    std::shared_ptr<IArchive> archive(new BruteforceArchive(objective_indices));
    std::shared_ptr<ISolutionInitializer> initializer(new CategoricalProbabilisticallyCompleteInitializer());
    std::shared_ptr<LinkageMetric> metric(new NMI());
    std::shared_ptr<FoSLearner> foslearner(new CategoricalLinkageTree(metric));

    std::shared_ptr<IPerformanceCriterion> performance_criterion(
        new DominationObjectiveAcceptanceCriterion(objective_indices));
    //
    Rng rng(42);
    pop->registerGlobalData(rng);
    std::shared_ptr<ObjectiveFunction> onemax(new OneMax(l, 0));
    std::shared_ptr<ObjectiveFunction> zeromax(new ZeroMax(l, 1));
    Compose compose({onemax, zeromax});
    compose.setPopulation(pop);

    pop->registerGlobalData(GObjectiveFunction(&compose));

    MO_GOMEA gomea(population_size,
                   number_of_clusters,
                   objective_indices,
                   initializer,
                   foslearner,
                   performance_criterion,
                   archive);
    gomea.setPopulation(pop);
    compose.registerData();
    gomea.registerData();
    compose.afterRegisterData();
    gomea.afterRegisterData();

    gomea.step();
    gomea.step();
    gomea.step();
    gomea.step();
    gomea.step();
}

TEST_CASE("Integration Test: Kernel-GOMEA on OneMax vs ZeroMax")
{
    std::shared_ptr<Population> pop = std::make_shared<Population>();

    size_t l = 100;
    size_t population_size = 16;
    size_t number_of_clusters = 3;
    std::vector<size_t> objective_indices = {0, 1};

    std::shared_ptr<IArchive> archive(new BruteforceArchive(objective_indices));
    std::shared_ptr<ISolutionInitializer> initializer(new CategoricalProbabilisticallyCompleteInitializer());
    std::shared_ptr<LinkageMetric> metric(new NMI());
    std::shared_ptr<FoSLearner> foslearner(new CategoricalLinkageTree(metric));

    std::shared_ptr<IPerformanceCriterion> performance_criterion(
        new DominationObjectiveAcceptanceCriterion(objective_indices));
    //
    Rng rng(42);
    pop->registerGlobalData(rng);
    std::shared_ptr<ObjectiveFunction> onemax(new OneMax(l, 0));
    std::shared_ptr<ObjectiveFunction> zeromax(new ZeroMax(l, 1));
    Compose compose({onemax, zeromax});
    compose.setPopulation(pop);

    pop->registerGlobalData(GObjectiveFunction(&compose));

    KernelGOMEA gomea(population_size,
                   number_of_clusters,
                   objective_indices,
                   initializer,
                   foslearner,
                   performance_criterion,
                   archive);
    gomea.setPopulation(pop);
    compose.registerData();
    gomea.registerData();
    compose.afterRegisterData();
    gomea.afterRegisterData();

    gomea.step();
    gomea.step();
    gomea.step();
    gomea.step();
    gomea.step();
}

TEST_CASE("Integration Test: Scalarized Kernel-GOMEA on OneMax vs ZeroMax")
{
    std::shared_ptr<Population> pop = std::make_shared<Population>();

    size_t l = 100;
    size_t population_size = 16;
    size_t number_of_clusters = 3;
    std::vector<size_t> objective_indices = {0, 1};

    std::shared_ptr<IArchive> archive(new BruteforceArchive(objective_indices));
    std::shared_ptr<ISolutionInitializer> initializer(new CategoricalProbabilisticallyCompleteInitializer());
    std::shared_ptr<LinkageMetric> metric(new NMI());
    std::shared_ptr<FoSLearner> foslearner(new CategoricalLinkageTree(metric));
    std::shared_ptr<Scalarizer> scalarizer(new TschebysheffObjectiveScalarizer(objective_indices));
    
    std::shared_ptr<GOMEAPlugin> plugin(new HoangScalarizationScheme(scalarizer, objective_indices));
    std::shared_ptr<IPerformanceCriterion> performance_criterion(
        new ScalarizationAcceptanceCriterion(scalarizer));
    //
    Rng rng(42);
    pop->registerGlobalData(rng);
    std::shared_ptr<ObjectiveFunction> onemax(new OneMax(l, 0));
    std::shared_ptr<ObjectiveFunction> zeromax(new ZeroMax(l, 1));
    Compose compose({onemax, zeromax});
    compose.setPopulation(pop);

    pop->registerGlobalData(GObjectiveFunction(&compose));

    KernelGOMEA gomea(population_size,
                   number_of_clusters,
                   objective_indices,
                   initializer,
                   foslearner,
                   performance_criterion,
                   archive,
                   plugin);
    gomea.setPopulation(pop);
    compose.registerData();
    gomea.registerData();
    compose.afterRegisterData();
    gomea.afterRegisterData();

    gomea.step();
    gomea.step();
    gomea.step();
    gomea.step();
    gomea.step();
}