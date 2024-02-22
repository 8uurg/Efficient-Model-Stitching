#pragma once

#include <catch2/catch.hpp>
#include <catch2/trompeloeil.hpp>
#include "running.hpp"

class MockRunner : public trompeloeil::mock_interface<IRunner>
{
  public:
    IMPLEMENT_MOCK0(run);
    IMPLEMENT_MOCK0(step);
    
    IMPLEMENT_MOCK1(setPopulation);
    IMPLEMENT_MOCK0(registerData);
    IMPLEMENT_MOCK0(afterRegisterData);
};