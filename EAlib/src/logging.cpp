#include "logging.hpp"
#include <filesystem>
#include <iomanip>

// SequencedItemLogger
SequencedItemLogger::SequencedItemLogger(std::vector<std::shared_ptr<ItemLogger>> subloggers) : subloggers(subloggers)
{
}
void SequencedItemLogger::setPopulation(std::shared_ptr<Population> population)
{
    for (auto &sublogger : subloggers)
    {
        sublogger->setPopulation(population);
    }
}
void SequencedItemLogger::registerData()
{
    for (auto &sublogger : subloggers)
    {
        sublogger->registerData();
    }
}
void SequencedItemLogger::afterRegisterData()
{
    for (auto &sublogger : subloggers)
    {
        sublogger->afterRegisterData();
    }
}
void SequencedItemLogger::header(IMapper &mapper)
{
    for (auto &sublogger : subloggers)
    {
        sublogger->header(mapper);
    }
}
void SequencedItemLogger::log(IMapper &mapper, const Individual &i)
{
    for (auto &sublogger : subloggers)
    {
        sublogger->log(mapper, i);
    }
}

std::shared_ptr<SequencedItemLogger> SequencedItemLogger::shared(std::vector<std::shared_ptr<ItemLogger>> subloggers)
{
    return std::make_shared<SequencedItemLogger>(subloggers);
}

// GenotypeCategoricalLogger
void GenotypeCategoricalLogger::afterRegisterData()
{
    Population &pop = *this->population;
    tggc.emplace(pop.getDataContainer<GenotypeCategorical>());
}
void GenotypeCategoricalLogger::header(IMapper &mapper)
{
    mapper << "genotype (categorical)";
}
void GenotypeCategoricalLogger::log(IMapper &mapper, const Individual &i)
{
    GenotypeCategorical &gc = tggc->getData(i);
    std::stringstream ss;
    for (size_t idx = 0; idx < gc.genotype.size(); ++idx)
    {
        if (idx != 0)
            ss << " ";
        ss << static_cast<long>(gc.genotype[idx]);
    }
    mapper << ss.str();
}

std::shared_ptr<GenotypeCategoricalLogger> GenotypeCategoricalLogger::shared()
{
    return std::make_shared<GenotypeCategoricalLogger>();
}

// ObjectiveLogger
void ObjectiveLogger::afterRegisterData()
{
    Population &pop = *this->population;
    tgo.emplace(pop.getDataContainer<Objective>());
}
void ObjectiveLogger::header(IMapper &mapper)
{
    mapper << "objectives";
}
void ObjectiveLogger::log(IMapper &mapper, const Individual &i)
{
    Objective &o = tgo->getData(i);
    std::stringstream ss;
    for (size_t idx = 0; idx < o.objectives.size(); ++idx)
    {
        if (idx != 0)
            ss << " ";
        ss << std::setprecision(6) << static_cast<double>(o.objectives[idx]);
    }
    mapper << ss.str();
}

std::shared_ptr<ObjectiveLogger> ObjectiveLogger::shared()
{
    return std::make_shared<ObjectiveLogger>();
}

// NumEvaluationsLogger
NumEvaluationsLogger::NumEvaluationsLogger(std::shared_ptr<Limiter> limiter) : limiter(limiter)
{
}
std::shared_ptr<NumEvaluationsLogger> NumEvaluationsLogger::shared(std::shared_ptr<Limiter> limiter)
{
    return std::make_shared<NumEvaluationsLogger>(limiter);
}

void NumEvaluationsLogger::header(IMapper &mapper)
{
    mapper << "#evaluations";
}
void NumEvaluationsLogger::log(IMapper &mapper, const Individual &)
{
    mapper << std::to_string(limiter->get_num_evaluations());
}
// NumWallTimeLogger
WallTimeLogger::WallTimeLogger(std::shared_ptr<Limiter> limiter) : limiter(limiter)
{
}
std::shared_ptr<WallTimeLogger> WallTimeLogger::shared(std::shared_ptr<Limiter> limiter)
{
    return std::make_shared<WallTimeLogger>(limiter);
}

void WallTimeLogger::header(IMapper &mapper)
{
    mapper << "wall time (ms)";
}
void WallTimeLogger::log(IMapper &mapper, const Individual &)
{
    mapper << std::to_string(limiter->get_time_spent_ms());
}
// SolutionIndexLogger
std::shared_ptr<SolutionIndexLogger> SolutionIndexLogger::shared()
{
    return std::make_shared<SolutionIndexLogger>();
}
void SolutionIndexLogger::header(IMapper &mapper)
{
    mapper << "solution index";
}
void SolutionIndexLogger::log(IMapper &mapper, const Individual &i)
{
    mapper << std::to_string(i.i);
}


// CSVLogger
CSVLogger::CSVLogger(std::filesystem::path out_path, std::shared_ptr<ItemLogger> item_logger) :
    out_path(out_path), item_logger(item_logger)
{
    if(std::filesystem::exists(out_path))
    {
        std::filesystem::remove(out_path);
    }
    header();
}
void CSVLogger::registerData()
{
    item_logger->registerData();
}
void CSVLogger::afterRegisterData()
{
    item_logger->afterRegisterData();
}
void CSVLogger::setPopulation(std::shared_ptr<Population> population)
{
    item_logger->setPopulation(population);
}
void CSVLogger::header()
{
    std::ofstream out(out_path, std::ios::app);
    CSVWriter mapper(out);

    mapper.start_type();
    item_logger->header(mapper);
    mapper.end_type();
}
void CSVLogger::log(const Individual &i)
{
    std::ofstream out(out_path, std::ios::app);
    CSVWriter mapper(out);

    mapper.start_record();
    item_logger->log(mapper, i);
    mapper.end_record();
}

std::shared_ptr<CSVLogger> CSVLogger::shared(std::filesystem::path out_path, std::shared_ptr<ItemLogger> item_logger)
{
    return std::make_shared<CSVLogger>(out_path, item_logger);
}

// 
