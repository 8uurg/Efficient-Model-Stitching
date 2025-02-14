//  DAEDALUS – Distributed and Automated Evolutionary Deep Architecture Learning with Unprecedented Scalability
// 
// This research code was developed as part of the research programme Open Technology Programme with project number 18373, which was financed by the Dutch Research Council (NWO), Elekta, and Ortec Logiqcare.
// 
// Project leaders: Peter A.N. Bosman, Tanja Alderliesten
// Researchers: Alex Chebykin, Arthur Guijt, Vangelis Kostoulas
// Main code developer: Arthur Guijt

#pragma once
// This file contains an interface & corresponding implementations for multi-objective archives.

#include <algorithm>
#include <limits>
#include <optional>

#include "base.hpp"
#include "logging.hpp"

struct Archived
{
    bool added;
    bool dominated;
    size_t ordinal;
    std::vector<size_t> ordinals_removed;
};

class IArchive : public IDataUser
{
  public:
    virtual Archived try_add(Individual candidate) = 0;

    virtual std::vector<Individual> &get_archived() = 0;
};

class BruteforceArchive : public IArchive
{
  public:
    BruteforceArchive(std::vector<size_t> objective_indices, std::optional<std::vector<double>> threshold = std::nullopt);

    Archived try_add(Individual candidate) override;

    std::vector<Individual> &get_archived() override;

    void set_threshold(size_t objective_idx, double max_value);
    std::vector<size_t> filter_threshold(size_t objective_idx, double max_value);

    void do_cache();

  private:
    struct Cache
    {
        TypedGetter<Objective> og;
    };
    std::optional<Cache> cache;

    std::vector<Individual> archive;
    const std::vector<size_t> objective_indices;
    std::optional<std::vector<double>> threshold;

    // Insertion & removal bookkeeping for logs.
    std::vector<size_t> archive_ord;
    size_t ordinal = 0;
};

class ArchivedLogger : public ItemLogger
{
  private:
    std::optional<std::reference_wrapper<Archived>> archived;

  public:
    static std::shared_ptr<ArchivedLogger> shared();

    void header(IMapper &mapper);
    void log(IMapper &mapper, const Individual &i);

    void setArchived(Archived &archived);
};

/**
 * @brief A wrapper that performs logging calls upon successful insertion around any archive .
 */
class LoggingArchive : public IArchive
{
  private:
    std::shared_ptr<IArchive> archive;
    std::shared_ptr<BaseLogger> logger;
    std::optional<std::shared_ptr<ArchivedLogger>> archive_log;

  public:
    LoggingArchive(std::shared_ptr<IArchive> archive,
                   std::shared_ptr<BaseLogger> logger,
                   std::optional<std::shared_ptr<ArchivedLogger>> archive_log = std::nullopt);

    void setPopulation(std::shared_ptr<Population> population) override;
    void registerData() override;
    void afterRegisterData() override;

    Archived try_add(Individual candidate) override;

    std::vector<Individual> &get_archived() override;
};