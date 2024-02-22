#pragma once
#include <catch2/catch.hpp>
#include <catch2/trompeloeil.hpp>

#include "archive.hpp"

class MockArchive : public trompeloeil::mock_interface<IArchive>
{
  public:
    IMPLEMENT_MOCK1(try_add);
    IMPLEMENT_MOCK0(get_archived);
    IMPLEMENT_MOCK1(setPopulation);
    IMPLEMENT_MOCK0(registerData);
    IMPLEMENT_MOCK0(afterRegisterData);
};