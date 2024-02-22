#include "archive.hpp"
#include <catch2/catch.hpp>
#include <random>

TEST_CASE("BruteforceArchive")
{
    std::shared_ptr<Population> pop = std::make_shared<Population>();
    pop->registerData<Objective>();
    BruteforceArchive archive({0, 1});
    archive.setPopulation(pop);

    Individual ia = pop->newIndividual();
    Objective &ia_o = pop->getData<Objective>(ia);
    ia_o.objectives = {-1, 0};

    auto r = archive.try_add(ia);
    REQUIRE(r.added == true);
    REQUIRE(r.dominated == false);
    REQUIRE(archive.get_archived().size() == 1);

    r = archive.try_add(ia);
    REQUIRE(r.added == false);
    REQUIRE(r.dominated == false);
    REQUIRE(archive.get_archived().size() == 1);

    ia_o = pop->getData<Objective>(ia);
    ia_o.objectives = {0, 0};

    r = archive.try_add(ia);
    REQUIRE(r.added == false);
    REQUIRE(r.dominated == true);
    REQUIRE(archive.get_archived().size() == 1);

    ia_o = pop->getData<Objective>(ia);
    ia_o.objectives = {-2, 0};

    r = archive.try_add(ia);
    REQUIRE(r.added == true);
    REQUIRE(r.dominated == false);
    REQUIRE(archive.get_archived().size() == 1);

    ia_o = pop->getData<Objective>(ia);
    ia_o.objectives = {0, -1};

    r = archive.try_add(ia);
    REQUIRE(r.added == true);
    REQUIRE(r.dominated == false);
    REQUIRE(archive.get_archived().size() == 2);

    ia_o = pop->getData<Objective>(ia);
    ia_o.objectives = {-1, -0.5};

    r = archive.try_add(ia);
    REQUIRE(r.added == true);
    REQUIRE(r.dominated == false);
    REQUIRE(archive.get_archived().size() == 3);

    ia_o = pop->getData<Objective>(ia);
    ia_o.objectives = {-3, -3};
    r = archive.try_add(ia);
    REQUIRE(r.added == true);
    REQUIRE(r.dominated == false);
    REQUIRE(archive.get_archived().size() == 1);
}

TEST_CASE("BruteforceArchive - threshold")
{
    std::shared_ptr<Population> pop = std::make_shared<Population>();
    pop->registerData<Objective>();
    BruteforceArchive archive({0, 1});
    archive.setPopulation(pop);

    Individual ia = pop->newIndividual();
    Objective &ia_o = pop->getData<Objective>(ia);
    ia_o.objectives = {-1, 1};
    auto r = archive.try_add(ia);
    ia_o.objectives = {0, 0};
    r = archive.try_add(ia);
    ia_o.objectives = {1, -1};
    r = archive.try_add(ia);

    REQUIRE(archive.get_archived().size() == 3);
    // 3 + 1 individual above.
    REQUIRE(pop->active() == 4);

    // Note - ordinals are one-indexed.

    SECTION("Filtering with threshold 2 on objective 0 should remove nothing")
    {
        auto removed = archive.filter_threshold(0, 2.0);
        REQUIRE(removed.size() == 0);
        REQUIRE(archive.get_archived().size() == 3);
        // 3 + 1 individual above.
        REQUIRE(pop->active() == 4);
    }
    SECTION("Set threshold with threshold 2 on objective 0 should remove nothing")
    {
        archive.set_threshold(0, 2.0);
        REQUIRE(archive.get_archived().size() == 3);
        // 3 + 1 individual above.
        REQUIRE(pop->active() == 4);
    }
    SECTION("Set threshold with threshold 2 on objective 0 should cause solutions with objective 0 > 2.0 to be rejected")
    {
        archive.set_threshold(0, 2.0);
        REQUIRE(archive.get_archived().size() == 3);
        // Even if the second objective is very high!
        ia_o.objectives = {3.0, -10};
        r = archive.try_add(ia);
        REQUIRE_FALSE(r.added);
        REQUIRE(archive.get_archived().size() == 3);
        // 3 + 1 individual above.
        REQUIRE(pop->active() == 4);
    }
    SECTION("Set threshold with threshold 2 on objective 0 should cause solutions with objective 0 < 2.0 to still be considered")
    {
        archive.set_threshold(0, 2.0);
        REQUIRE(archive.get_archived().size() == 3);
        ia_o.objectives = {-2.0, 2};
        r = archive.try_add(ia);
        REQUIRE(r.added);
        REQUIRE(archive.get_archived().size() == 4);
        // 4 + 1 individual above.
        REQUIRE(pop->active() == 5);
    }
    SECTION("Filtering with threshold 1 on objective 0 should remove nothing")
    {
        auto removed = archive.filter_threshold(0, 1.0);
        REQUIRE(removed.size() == 0);
        REQUIRE(archive.get_archived().size() == 3);
        // 3 + 1 individual above.
        REQUIRE(pop->active() == 4);
    }
    SECTION("Filtering with threshold 0 on objective 0 should remove ordinal 2")
    {
        auto removed = archive.filter_threshold(0, 0.0);
        REQUIRE(removed.size() == 1);
        REQUIRE(removed[0] == 3);
        REQUIRE(archive.get_archived().size() == 2);
        // 2 + 1 individual above.
        REQUIRE(pop->active() == 3);
    }
    SECTION("Filtering with threshold -1 on objective 0 should remove ordinal 1 and 2")
    {
        auto removed = archive.filter_threshold(0, -1.0);
        REQUIRE(removed.size() == 2);
        REQUIRE((removed[0] == 3 || removed[0] == 2));
        REQUIRE((removed[1] == 3 || removed[1] == 2));
        REQUIRE(archive.get_archived().size() == 1);
        // 1 + 1 individual above.
        REQUIRE(pop->active() == 2);
    }
    SECTION("Filtering with threshold -1 on objective 1 should remove ordinal 0 and 1")
    {
        auto removed = archive.filter_threshold(1, -1.0);
        REQUIRE(removed.size() == 2);
        REQUIRE((removed[0] == 1 || removed[0] == 2));
        REQUIRE((removed[1] == 1 || removed[1] == 2));
        REQUIRE(archive.get_archived().size() == 1);
        // 1 + 1 individual above.
        REQUIRE(pop->active() == 2);
    }
}
