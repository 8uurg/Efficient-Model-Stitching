#pragma once

#include <catch2/catch.hpp>
#include <catch2/trompeloeil.hpp>

#include "decentralized.hpp"

class MockTaggedEventQueue : public trompeloeil::mock_interface<ITaggedEventQueue>
{
    IMPLEMENT_MOCK2(next);
    IMPLEMENT_MOCK1(setPopulation);
    IMPLEMENT_MOCK0(registerData);
    IMPLEMENT_MOCK0(afterRegisterData);
};