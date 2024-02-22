#pragma once
#include <catch2/catch.hpp>
#include <catch2/trompeloeil.hpp>

class GenericMockFunction
{
  public:
    MAKE_MOCK0(mock_function, void());
};