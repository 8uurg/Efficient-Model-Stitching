#pragma once

#include <catch2/catch.hpp>
#include <catch2/trompeloeil.hpp>
#include "gomea.hpp"
#include "trompeloeil.hpp"

class MockSamplingDistribution : public trompeloeil::mock_interface<ISamplingDistribution>
{
  public:
    IMPLEMENT_MOCK2(apply_resample);
};

class MockFoSLearner : public trompeloeil::mock_interface<FoSLearner>
{
  public:
    IMPLEMENT_MOCK1(learnFoS);
    IMPLEMENT_MOCK0(getFoS);
    IMPLEMENT_MOCK0(cloned_ptr);
    IMPLEMENT_MOCK1(setPopulation);
    IMPLEMENT_MOCK0(registerData);
    IMPLEMENT_MOCK0(afterRegisterData);
};