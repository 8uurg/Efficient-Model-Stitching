#include "base.hpp"
#include "decentralized.hpp"
#include "pybind11/pytypes.h"

class PythonAsyncIOEQ : public ITaggedEventQueue
{
    py::object asyncio;
    std::vector<size_t> tag;
    py::object flag;
    py::object event_loop;

  public:
  
    PythonAsyncIOEQ(py::object _event_loop)
    {
        asyncio = py::module::import("asyncio");
        flag = asyncio.attr("Event")();

        if (_event_loop.is_none()) {
            event_loop = asyncio.attr("get_event_loop")();
        } else {
            event_loop = _event_loop;
        }
    }

    void provide_tag(size_t tag)
    {
        taggy->num_pending_tags--;
        this->tag.push_back(tag);
        // Inform the event loop that we are done!
        flag.attr("set")();
    }

    bool next(Scheduler & /* wd */, size_t *tag) override
    {
        // Wait until a tag (or multiple) are provided.
        // If there are no more pending tags: abort.
        if (this->tag.size() == 0 && taggy->num_pending_tags > 0)
        {
            // Run event loop until a tag is provided.
            event_loop.attr("run_until_complete")(flag.attr("wait")());
            // Clear, so that we can reuse the flag.
            flag.attr("clear")();
        }

        // Events, like CTRL+C, will also stop the aforementioned step, so we haven't
        // always gotten a new tag.
        if (this->tag.size() > 0)
        {
            *tag = this->tag.back();
            this->tag.pop_back();
            return true;
        }
        else
        {
            return false;
        }
    }

    class TagCompleter
    {
    public:
        PythonAsyncIOEQ* paiqeq;
        size_t num_pending_tags = 0;

        TagCompleter(PythonAsyncIOEQ* paiqeq) : paiqeq(paiqeq) {}

        void provide_tag(size_t tag)
        {
            paiqeq->provide_tag(tag);
        }

        py::object get_event_loop() {
            return paiqeq->event_loop;
        }
    };

    
  private:
    std::shared_ptr<PythonAsyncIOEQ::TagCompleter> taggy;

  public:

    void registerData() override
    {
        population->registerGlobalData(TagCompleter(this));
        taggy = population->getGlobalData<TagCompleter>();
    }

};

class PyAsyncPending : public IResumable
{
    size_t parent_tag;

  public:
    py::object a;
    py::object raise_py_exception;

    PyAsyncPending(size_t parent_tag, py::object raise_py_exception) : parent_tag(parent_tag), raise_py_exception(raise_py_exception)
    {
    }

    bool resume(Scheduler &wd, std::unique_ptr<IResumable> &)
    {
        // We resume - to tag this task as complete.
        wd.complete_tag(parent_tag);

        // Check if this task threw an exception
        auto maybe_py_exception = a.attr("exception")();
        // And raise it if it exists now - so that pybind may deal with the conversion
        // and translation related complexity.
        if (! maybe_py_exception.is_none())
            raise_py_exception(maybe_py_exception);
        
        // This task has no follow-up.
        return false;
    }
};

class PyAsyncObjectiveFunction : public ObjectiveFunction
{
    std::shared_ptr<Scheduler> wd;
    std::shared_ptr<ObjectiveFunction> problem_template;
    py::object async_evaluation_function;
    std::shared_ptr<PythonAsyncIOEQ::TagCompleter> paioqtc;
    py::object async_wrapper;
    py::object raise_py_exception;

  public:
    PyAsyncObjectiveFunction(std::shared_ptr<Scheduler> wd,
                             std::shared_ptr<ObjectiveFunction> problem_template,
                             py::object async_evaluation_function) :
        wd(wd), problem_template(problem_template), async_evaluation_function(async_evaluation_function)
    {
        auto ealib = py::module::import("ealib");
        async_wrapper = ealib.attr("wrap_tag_completion");
        raise_py_exception = ealib.attr("raise_py_exception");
    }
    void evaluate(Individual i) override
    {
        // Some complexity in interacting with async functions here!
        // async_evaluation_function requires a wrapper that we can schedule, that
        // forwards the completion of the tag as an event to the inner event loop.
        t_assert(wd->tag_stack.size() > 0, "Callback is required for PyAsyncObjectiveFunction to work.")
            size_t parent_tag = wd->tag_stack.back();
        wd->tag_stack.pop_back();

        auto r = std::make_unique<PyAsyncPending>(parent_tag, raise_py_exception);
        auto rp = r.get();
        size_t tag = wd->get_tag(std::move(r));

        paioqtc->num_pending_tags++;

        // Python function has signature async ... (Population, Individual)
        auto wrapped = async_wrapper(paioqtc, tag, async_evaluation_function, population, i);
        auto loop = paioqtc->get_event_loop();
        rp->a = loop.attr("create_task")(wrapped());
    }
    ObjectiveFunction *clone() override
    {
        return new PyAsyncObjectiveFunction(wd, problem_template, async_evaluation_function);
    }

    void setPopulation(std::shared_ptr<Population> population) override
    {
        // Note - we do not set the population of wd, as this should be done by another class.
        this->population = population;
        if (problem_template != nullptr)
            problem_template->setPopulation(population);
    }
    void registerData() override
    {
        // Note - we do not registerData for wd, as this should be done by another class.
        if (problem_template != nullptr)
            problem_template->registerData();
    }
    void afterRegisterData() override
    {
        // Note - we do not afterRegisterData for wd, as this should be done by another class.
        t_assert(population->isGlobalRegistered<PythonAsyncIOEQ::TagCompleter>(), "PythonAsyncIOEQ::TagCompleter should be registered.");
        paioqtc = population->getGlobalData<PythonAsyncIOEQ::TagCompleter>();

        if (problem_template != nullptr)
            problem_template->afterRegisterData();
    }
};
