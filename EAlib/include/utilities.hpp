//  DAEDALUS – Distributed and Automated Evolutionary Deep Architecture Learning with Unprecedented Scalability
// 
// This research code was developed as part of the research programme Open Technology Programme with project number 18373, which was financed by the Dutch Research Council (NWO), Elekta, and Ortec Logiqcare.
// 
// Project leaders: Peter A.N. Bosman, Tanja Alderliesten
// Researchers: Alex Chebykin, Arthur Guijt, Vangelis Kostoulas
// Main code developer: Arthur Guijt

#pragma once

#include <cmath>
#include <cppassert.h>
#include <iostream>
#include <limits>
#include <numeric>
#include <regex>
#include <variant>

class IMapper
{
  public:
    virtual ~IMapper() = default;
    // For hierarchical blocks -- if supported
    virtual void start_block(std::string){};
    virtual void end_block(){};
    // Type header
    virtual void start_type(){};
    virtual void end_type(){};
    // Record based separator
    virtual void start_record(){};
    virtual void end_record(){};

    virtual IMapper &operator<<(std::string)
    {
        return *this;
    };
};

class CSVWriter : public IMapper
{
  public:
    CSVWriter(std::ostream &out) : out(out){};

    CSVWriter &operator<<(std::string ss)
    {
        if (!first)
            out << sep;

        // Double up "" to escape
        ss = std::regex_replace(ss, std::regex("\""), "\"\"");
        bool requires_escaping = std::regex_search(ss, std::regex("[,\"]"));

        if (requires_escaping)
            out << '"';
        out << ss;
        if (requires_escaping)
            out << '"';

        first = false;
        return *this;
    }
    void end_record()
    {
        first = true;
        out << record_sep;
    }
    virtual void end_type()
    {
        first = true;
        out << record_sep;
    }

  private:
    bool first = true;
    char sep = ',';
    char record_sep = '\n';
    std::ostream &out;
};

template <typename T> class Matrix
{
  public:
    Matrix(T d, size_t w, size_t h) : w(w), h(h)
    {
        size_t n = w * h;
        elements.resize(n);
        std::fill(elements.begin(), elements.end(), d);
    };

    T &operator[](std::pair<size_t, size_t> index)
    {
        size_t row = std::get<0>(index);
        size_t col = std::get<1>(index);
        return elements[getIndexAt(row, col)];
    }

    T get(size_t row, size_t col)
    {
        return elements[getIndexAt(row, col)];
    };
    void set(size_t row, size_t col, T v)
    {
        elements[getIndexAt(row, col)] = v;
    }

    size_t getWidth()
    {
        return w;
    }

    size_t getHeight()
    {
        return h;
    }

    void swap(Matrix<T> &m)
    {
        m.elements.swap(this->elements);
        std::swap(m.h, this->h);
        std::swap(m.w, this->w);
    }

  private:
    std::vector<T> elements;
    size_t w;
    size_t h;

    size_t getIndexAt(size_t row, size_t col)
    {
        if (row >= h || col >= w) {
            std::cerr << "got row " << row << " which should be < " << h << " and col " << col << " which should be < " << w << ".";
        }
        t_assert(row < h && col < w, "i and j should be in bounds");
        // i is row, j is column
        // w is row width
        return row * w + col;
    }
};

template <typename T> class SymMatrix
{
  public:
    SymMatrix(std::vector<std::vector<T>> vector)
    {
        if (vector.size() == 0)
        {
            // Edge case...
            this->n = 0;
            elements.resize(this->n);
            return;
        }
        n = vector[0].size();
        elements.resize(0);
        elements.reserve(getIndexAt(n - 1, n - 1) + 1);
        for (size_t row_idx = 0; row_idx < vector.size(); ++row_idx)
        {
            t_assert(vector[row_idx].size() == n - row_idx,
                     "symmetric matrix initializer requires only the upper triangular elements to be specified.");
            elements.insert(elements.end(), vector[row_idx].begin(), vector[row_idx].end());
        }
    };
    SymMatrix(T d, size_t n) : n(n)
    {
        elements.resize(getIndexAt(n - 1, n - 1) + 1);
        std::fill(elements.begin(), elements.end(), d);
    };
    T get(size_t i, size_t j)
    {
        return elements[getIndexAt(i, j)];
    };
    T &operator[](std::pair<size_t, size_t> index)
    {
        size_t row = std::get<0>(index);
        size_t col = std::get<1>(index);
        return elements[getIndexAt(row, col)];
    }
    void set(size_t i, size_t j, T v)
    {
        elements[getIndexAt(i, j)] = v;
    }
    size_t getSize()
    {
        return n;
    }

  private:
    std::vector<T> elements;
    size_t n;

    size_t getIndexAt(size_t i, size_t j)
    {
        size_t row = std::min(i, j);
        size_t col = std::max(i, j);
        t_assert(row < n && col < n, "i and j should be in bounds");

        size_t previous = (row * (2 * n + 1 - row)) / 2;

        return previous + col - row;
    }
};

template <typename L>
std::vector<size_t> greedyScatteredSubsetSelection(L &&distance, size_t pool_size, size_t k, size_t start)
{
    t_assert(k <= pool_size, "Can select at most pool_size items.");

    if (k <= 1)
        return {start};

    // Rather than selecting the first item ourselves, you get to provide it instead. :)

    // Indices left: skipping start.
    std::vector<size_t> indices_left(pool_size - 1);
    std::iota(indices_left.begin(), indices_left.begin() + static_cast<long>(start), 0);
    std::iota(indices_left.begin() + static_cast<long>(start), indices_left.end(), start + 1);

    // Compute initial distances: the distances to start.
    std::vector<double> distances(pool_size);
    size_t next_ii = 0;
    size_t next = 0;
    double d = -std::numeric_limits<double>::infinity();
    for (size_t ii = 0; ii < indices_left.size(); ++ii)
    {
        size_t i = indices_left[ii];
        distances[i] = distance(start, i);
        if (distances[i] > d)
        {
            d = distances[i];
            next_ii = ii;
            next = i;
        }
    }
    // Remove furthest element
    std::swap(indices_left[next_ii], indices_left.back());
    indices_left.pop_back();

    std::vector<size_t> subset = {start, next};
    size_t current = next;

    while (subset.size() < k)
    {
        d = -std::numeric_limits<double>::infinity();
        for (size_t ii = 0; ii < indices_left.size(); ++ii)
        {
            size_t i = indices_left[ii];
            distances[i] = std::min(distances[i], distance(current, i));
            if (distances[i] > d)
            {
                d = distances[i];
                next_ii = ii;
                next = i;
            }
        }
        current = next;
        subset.push_back(next);
    }

    return subset;
}

template <typename T> double euclidean_distance(T *a, T *b, size_t l)
{
    double d = 0;
    for (size_t idx = 0; idx < l; ++idx)
    {
        double s = (a[idx] - b[idx]);
        d += s * s;
    }
    return std::sqrt(d);
}

template <typename T> int hamming_distance(T *a, T *b, size_t l)
{
    int d = 0;
    for (size_t idx = 0; idx < l; ++idx)
    {
        d += !(a[idx] == b[idx]);
    }
    return d;
}

template <typename T> using APtr = std::variant<std::unique_ptr<T>, T *>;

template <typename T> T *as_ptr(APtr<T> &ptr)
{
    if (std::holds_alternative<std::unique_ptr<T>>(ptr))
        return std::get<std::unique_ptr<T>>(ptr).get();
    else
        return std::get<T *>(ptr);
}