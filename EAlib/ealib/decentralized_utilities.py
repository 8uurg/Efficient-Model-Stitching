from ._ealib import TagCompleter, Population, Individual
import asyncio

def wrap_tag_completion(q: TagCompleter, tag: int, fn, *args):
    async def wrapper():
        nonlocal q
        # Similar to the note above - if we immidiately run this task, Python's event loop will starve.
        # As these tasks should generally be inexpensive - do them - so that parallel evaluations may take
        # place.
        await asyncio.sleep(0)
        try:
            await fn(*args)
        finally:
            # Now - once we provide the tag - the EA resumes immidiately, stopping
            # us from processing other completed evaluations and blocking us from evaluating
            # more solutions. In general, Python's event loop will starve.
            # Insert a breakpoint before we return here as to allow other steps to complete too -
            # and deferring the restart until the end.
            await asyncio.sleep(0)
            # Provide tag, even if exception was thrown, so that the EA resumes
            # PyAsyncPending needs to check if the task raised an exception and throw it accordingly.
            q.provide_tag(tag)
    return wrapper

def raise_py_exception(o):
    raise o