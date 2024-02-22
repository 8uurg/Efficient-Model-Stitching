#include "archive.hpp"
#include "cppassert.h"

// BruteforceArchive
BruteforceArchive::BruteforceArchive(std::vector<size_t> objective_indices,
                                     std::optional<std::vector<double>> threshold) : objective_indices(objective_indices), threshold(threshold)
{
}

Archived BruteforceArchive::try_add(Individual candidate)
{
    do_cache();
    Population &population = *this->population;

    bool dominated = false;
    bool added = true;
    std::vector<size_t> ordinals_removed;

    Objective &candidate_o = cache->og.getData(candidate);
    for (size_t i = 0; i < objective_indices.size(); ++i)
    {
        auto &oi = objective_indices[i];
        if (candidate_o.objectives.size() <= oi)
        {
            std::cerr << "Got solution with " << candidate_o.objectives.size() << " objectives specified. Expected at least " << oi << "." << std::endl;
        }
        t_assert(candidate_o.objectives.size() > oi, "Solution to be added to archive should be evaluated.");

        if (threshold.has_value())
        {
            // Don't archive above threshold.
            auto &thx = *threshold;
            if (thx.size() > i && candidate_o.objectives[oi] > thx[i])
            {
                added = false;
                dominated = true;
                return Archived{added, dominated, ordinal, std::move(ordinals_removed)};
            }
        }
    }

    for (size_t archive_idx = 0; archive_idx < archive.size(); ++archive_idx)
    {
        Individual &i = archive[archive_idx];
        bool i_has_better_objective_value = false;
        bool candidate_has_better_objective_value = false;

        Objective &i_o = cache->og.getData(i);
        for (size_t io = 0; io < objective_indices.size(); ++io)
        {
            size_t o = objective_indices[io];
            if (i_o.objectives[o] < candidate_o.objectives[o])
                i_has_better_objective_value = true;
            if (i_o.objectives[o] > candidate_o.objectives[o])
                candidate_has_better_objective_value = true;
        }

        if (!i_has_better_objective_value && !candidate_has_better_objective_value)
        {
            // Solutions are equal
            added = false;
            break;
        }
        else if (i_has_better_objective_value && candidate_has_better_objective_value)
        {
            // Solution is not dominated, but does not dominate the other either.
        }
        else if (i_has_better_objective_value)
        {
            // Candidate is dominated -- we can stop.
            dominated = true;
            added = false;
            break;
        }
        else if (candidate_has_better_objective_value)
        {
            // Candidate dominates i -- i can be removed.
            std::swap(archive[archive_idx], archive.back());
            std::swap(archive_ord[archive_idx], archive_ord.back());
            population.dropIndividual(archive.back());
            ordinals_removed.push_back(archive_ord.back());
            archive.pop_back();
            archive_ord.pop_back();

            // Current index is now a new element, so revert a step :)
            archive_idx--;
        }
    }

    if (added)
    {
        Individual archived_candidate = population.newIndividual();
        population.copyIndividual(candidate, archived_candidate);
        archive_ord.push_back(++ordinal);
        archive.push_back(archived_candidate);
    }

    return Archived{added, dominated, ordinal, std::move(ordinals_removed)};
}

std::vector<Individual> &BruteforceArchive::get_archived()
{
    return archive;
}


std::vector<size_t> BruteforceArchive::filter_threshold(size_t objective_idx, double max_value)
{
    do_cache();
    std::vector<size_t> ordinals_removed;
    
    for (size_t archive_idx = 0; archive_idx < archive.size(); ++archive_idx)
    {

        Objective &s_o = cache->og.getData(archive[archive_idx]);
        if (// If unevaluated - does not meet threshold
            s_o.objectives.size() <= objective_idx ||
            // If worse than threshold, prune too.
            s_o.objectives[objective_idx] > max_value)
        {
            // Candidate dominates i -- i can be removed.
            std::swap(archive[archive_idx], archive.back());
            std::swap(archive_ord[archive_idx], archive_ord.back());
            population->dropIndividual(archive.back());
            // Unlike the archive - we do not track things here.
            ordinals_removed.push_back(archive_ord.back());
            archive.pop_back();
            archive_ord.pop_back();

            // Current index is now a new element, so revert a step :)
            archive_idx--;
        }
    }
    return ordinals_removed;
}
void BruteforceArchive::set_threshold(size_t objective_idx, double max_value)
{
    auto objective_idx_pos = std::find(objective_indices.begin(), objective_indices.end(), objective_idx);
    t_assert(objective_idx_pos != objective_indices.end(),
             "Setting threshold only possible for indices tracked by this archive.");
    size_t pos = objective_idx_pos - objective_indices.begin();

    if (threshold.has_value())
    {
        auto &x = *threshold;
        if (x.size() <= pos)
        {
            x.resize(pos + 1);
        }
        x[pos] = max_value;
    }
    else
    {
        std::vector<double> x;
        x.resize(pos + 1);
        std::fill(x.begin(), x.end(), std::numeric_limits<double>::infinity());
        x[pos] = max_value;
        threshold.emplace(std::move(x));
    }
    // Also - filter the solutions already in the archive for good measure.
    filter_threshold(objective_idx, max_value);
}

void BruteforceArchive::do_cache()
{
    if (!cache.has_value())
    {
        Population &population = *this->population;
        auto og = population.getDataContainer<Objective>();
        cache.emplace(Cache{og});
    }
}

// ArchivedLogger
std::shared_ptr<ArchivedLogger> ArchivedLogger::shared()
{
    return std::make_shared<ArchivedLogger>();
}

void ArchivedLogger::header(IMapper &mapper)
{
    mapper << "archive ordinal"
           << "archive ordinals removed";
}
void ArchivedLogger::log(IMapper &mapper, const Individual &)
{
    if (!archived.has_value())
    {
        mapper << "missing"
               << "missing";
        return;
    }
    auto &archived_v = archived->get();
    mapper << std::to_string(archived_v.ordinal);

    std::stringstream ss;
    for (size_t idx = 0; idx < archived_v.ordinals_removed.size(); ++idx)
    {
        if (idx != 0)
            ss << " ";
        ss << archived_v.ordinals_removed[idx];
    }
    mapper << ss.str();

    archived.reset();
}
void ArchivedLogger::setArchived(Archived &archived)
{
    this->archived.emplace(archived);
}

// LoggingArchive
LoggingArchive::LoggingArchive(std::shared_ptr<IArchive> archive,
                               std::shared_ptr<BaseLogger> logger,
                               std::optional<std::shared_ptr<ArchivedLogger>> archive_log) :
    archive(archive), logger(logger), archive_log(archive_log)
{
}
void LoggingArchive::setPopulation(std::shared_ptr<Population> population)
{
    archive->setPopulation(population);
    logger->setPopulation(population);
}
void LoggingArchive::registerData()
{
    archive->registerData();
    logger->registerData();
}
void LoggingArchive::afterRegisterData()
{
    archive->afterRegisterData();
    logger->afterRegisterData();
}
Archived LoggingArchive::try_add(Individual candidate)
{
    Archived a = archive->try_add(candidate);

    if (a.added)
    {
        if (archive_log.has_value())
        {
            auto &al = *archive_log;
            al->setArchived(a);
        }
        logger->log(candidate);
    }

    return a;
}
std::vector<Individual> &LoggingArchive::get_archived()
{
    return archive->get_archived();
}
