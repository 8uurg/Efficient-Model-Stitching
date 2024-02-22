
#include <memory>

#include "sim.hpp"

#include <catch2/catch.hpp>

template<typename T>
std::unique_ptr<IResumableSimulated> lambda_to_resumable(T&& f)
{
    return std::make_unique<FunctionalSimResumable>([f](ISimulator &, double, std::unique_ptr<IResumableSimulated> &){
        f();
    });
}

TEST_CASE("Simulator step: in-order insertion")
{
    Simulator sim;
    bool hit0 = false;
    bool hit1 = false;

    sim.insert_event(lambda_to_resumable(
        [&hit0](){ hit0 = true; }
    ), 1.0, std::nullopt);
    sim.insert_event(lambda_to_resumable(
        [&hit1](){ hit1 = true; }
    ), 2.0, std::nullopt);
    sim.step();

    SECTION("first step")
    {
        CHECK(hit0);
        CHECK(!hit1);
        CHECK(sim.now() == 1.0);
    }

    sim.step();

    SECTION("second step")
    {
        
        CHECK(hit0);
        CHECK(hit1);
        CHECK(sim.now() == 2.0);
    }
}

TEST_CASE("Simulator step: out-of-order insertion")
{
    Simulator sim;
    bool hit0 = false;
    bool hit1 = false;

    sim.insert_event(lambda_to_resumable(
        [&hit1](){ hit1 = true; }
    ), 2.0, std::nullopt);
    sim.insert_event(lambda_to_resumable(
        [&hit0](){ hit0 = true; }
    ), 1.0, std::nullopt);

    sim.step();

    SECTION("first step")
    {
        CHECK(hit0);
        CHECK(!hit1);
        CHECK(sim.now() == 1.0);
    }

    sim.step();

    SECTION("second step")
    {
        
        CHECK(hit0);
        CHECK(hit1);
        CHECK(sim.now() == 2.0);
    }
}

TEST_CASE("Simulator step: time travel is disallowed")
{
    Simulator sim;
    bool hit0 = false;
    bool hit1 = false;
    bool hit2 = false;

    sim.insert_event(lambda_to_resumable(
        [&hit1](){ hit1 = true; }
    ), 2.0, std::nullopt);
    sim.insert_event(lambda_to_resumable(
        [&hit0](){ hit0 = true; }
    ), 1.0, std::nullopt);

    sim.step();

    SECTION("first step")
    {
        CHECK(hit0);
        CHECK(!hit2);
        CHECK(!hit1);
        CHECK(sim.now() == 1.0);
    }

    // Note: time travelling event!
    sim.insert_event(lambda_to_resumable(
        [&hit2](){ hit2 = true; }
    ), 0.0, std::nullopt);

    sim.step();

    SECTION("second step")
    {
        CHECK(hit0);
        CHECK(hit2);
        CHECK(!hit1);
        CHECK(sim.now() == 1.0);
    }

    sim.step();

    SECTION("third step")
    {
        CHECK(hit0);
        CHECK(hit2);
        CHECK(hit1);
        CHECK(sim.now() == 2.0);
    }
}