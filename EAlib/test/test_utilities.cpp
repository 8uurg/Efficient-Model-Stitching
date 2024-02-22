#include <catch2/catch.hpp>
#include <sstream>
#include <utilities.hpp>

TEST_CASE("CSVWriter", "[Writer]")
{
    std::stringstream oss;
    CSVWriter writer = CSVWriter(oss);

    SECTION("simple singular write")
    {
        writer << "b";
        REQUIRE_THAT(oss.str(), Catch::Matchers::Equals("b"));
    }

    SECTION("multiple writes are separated")
    {
        writer << "b"
               << "b";
        REQUIRE_THAT(oss.str(), Catch::Matchers::Equals("b,b"));
    }

    SECTION("strings with ',''s in them are quoted")
    {
        writer << "aaa,bbb";
        REQUIRE_THAT(oss.str(), Catch::Matchers::Equals("\"aaa,bbb\""));
    }

    SECTION("strings with '\"''s in them are quoted (to avoid ambiguity)")
    {
        writer << "\"";
        REQUIRE_THAT(oss.str(), Catch::Matchers::Equals("\"\"\"\""));
    }

    SECTION("ending a record should insert a newline")
    {
        writer.end_record();
        REQUIRE_THAT(oss.str(), Catch::Matchers::Equals("\n"));
    }

    SECTION("no separator should be inserted after ending a record")
    {
        writer << "b";
        writer.end_record();
        writer << "b";
        REQUIRE_THAT(oss.str(), Catch::Matchers::Equals("b\nb"));
    }
}

TEST_CASE("Matrix")
{
    auto m = Matrix(0, 10, 10);

    SECTION("should throw out of bounds if i >= 10")
    {
        REQUIRE_THROWS(m.get(10, 0));
    }
    SECTION("should throw out of bounds if j >= 10")
    {
        REQUIRE_THROWS(m.get(0, 10));
    }
    SECTION("diagonal should work")
    {
        m.set(0, 0, 10);
        REQUIRE(m.get(0, 0) == 10);
    }
    SECTION("off diagonal should work")
    {
        REQUIRE(m.get(0, 1) == 0);
        m.set(0, 1, 10);
        REQUIRE(m.get(0, 1) == 10);
    }
    SECTION("off diagonal should not enforce symmetry")
    {
        REQUIRE(m.get(1, 0) == 0);
        m.set(0, 1, 10);
        REQUIRE(m.get(1, 0) == 0);
        REQUIRE(m.get(0, 1) == 10);
    }
    SECTION("largest valid position")
    {
        REQUIRE(m.get(9, 9) == 0);
        m.set(9, 9, 10);
        REQUIRE(m.get(9, 9) == 10);
    }
}

TEST_CASE("SymMatrix")
{
    auto m = SymMatrix(0, 10);

    SECTION("should throw out of bounds if i >= 10")
    {
        REQUIRE_THROWS(m.get(10, 0));
    }
    SECTION("should throw out of bounds if j >= 10")
    {
        REQUIRE_THROWS(m.get(0, 10));
    }
    SECTION("diagonal should work")
    {
        m.set(0, 0, 10);
        REQUIRE(m.get(0, 0) == 10);
    }
    SECTION("off diagonal should work")
    {
        REQUIRE(m.get(0, 1) == 0);
        m.set(0, 1, 10);
        REQUIRE(m.get(0, 1) == 10);
    }
    SECTION("off diagonal should enforce symmetry")
    {
        REQUIRE(m.get(1, 0) == 0);
        m.set(0, 1, 10);
        REQUIRE(m.get(1, 0) == 10);
    }
    SECTION("largest valid position")
    {
        REQUIRE(m.get(9, 9) == 0);
        m.set(9, 9, 10);
        REQUIRE(m.get(9, 9) == 10);
    }
}

TEST_CASE("greedyScatteredSubsetSelection")
{
    SECTION("one point returns starting point")
    {
        auto result = greedyScatteredSubsetSelection([](int a, int b) { return static_cast<double>(std::abs(a - b)); }, 3, 1, 0);
        REQUIRE_THAT(result, Catch::Matchers::UnorderedEquals(std::vector<size_t>{0}));
    }
    SECTION("two points returns furthest")
    {
        auto result = greedyScatteredSubsetSelection([](int a, int b) { return static_cast<double>(std::abs(a - b)); }, 5, 2, 0);
        REQUIRE_THAT(result, Catch::Matchers::UnorderedEquals(std::vector<size_t>{0, 4}));
    }
    SECTION("three points")
    {
        auto result = greedyScatteredSubsetSelection([](int a, int b) { return static_cast<double>(std::abs(a - b)); }, 5, 3, 0);
        REQUIRE_THAT(result, Catch::Matchers::UnorderedEquals(std::vector<size_t>{0, 4, 2}));
    }
    SECTION("too many")
    {
        REQUIRE_THROWS(greedyScatteredSubsetSelection([](int a, int b) { return static_cast<double>(std::abs(a - b)); }, 3, 5, 0));
    }
}