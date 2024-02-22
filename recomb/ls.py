from copy import copy
from functools import partial
def enumerate_parallel_set(g, set_idx, parallel_sets, i, verbose=False):
    def call_funcs(lfn):
        for fn in lfn:
            fn()

    for (set_list, restore_list) in enumerate_parallel_set_recur(g, set_idx, parallel_sets, i, None, set(), verbose=verbose):
        yield (lambda: call_funcs(set_list)), (lambda: call_funcs(restore_list))

def enumerate_parallel_set_recur(g, set_idx, parallel_sets, i, current_set=None, unpickable: set=set(), ref_og=None, verbose=False):
    if current_set is None:
        # Initial case - current set is the parallel set of the index we are starting with.
        current_set = copy(parallel_sets[i])
        ref_og = g.vs[i]["og"]
        # filter current set based on a match
        current_set = {a for a in current_set if g.vs[a]["og"] == ref_og}
    else:
        # Otherwise, update the set of uncovered elements.
        current_set = current_set.intersection(parallel_sets[i])

    if len(current_set) == 0:
        if verbose: print(f"base case - no other choices necessary after picking {i}")
        # yield setter for configuring and unconfiguring i. No other configurations necessary
        # as there are no other branches.
        yield [partial(set_idx, i, 1)], [partial(set_idx, i, 0)]
        # return - as there are no more elements in the neighborhood.
        return

    # Obtain a fixed ordering of the set of leftover elements to be picked.
    ordering = list(current_set - unpickable)
    
    # Find current reverse cumulative intersection.
    # The intersection of sets picked so far provides knowledge of elements that may need
    # to be picked to cover all branches.
    # If we perform this operation cumulatively from the right the elements left in the
    # set allow us to identify necessary picks.
    # if we have the set with fixed ordering [ 1, 2, 3]
    # and the set corresponding here are 1 -> {0, 2, 3}, 2 -> {0, 1, 3}, 3 -> {0, 1, 2}
    # (note: the index itself is never contained within its own parallel set)
    # In this case the sequence of sets would be
    # [{}, {1}, {1, 2}]
    # as the only set that does not contain {1} is the set corresponding to {1}, 1 must be picked.
    cumulative_sets_rl = [None for _ in range(len(ordering))]
    cumulative_sets_rl[-1] = current_set.intersection(parallel_sets[ordering[-1]])
    required_right = {ordering[-1]}
    for i in range(len(ordering) - 1, 0, -1):
        el = ordering[i - 1]
        cumulative_sets_rl[i - 1] = cumulative_sets_rl[i].intersection(parallel_sets[el])
        # note - if an ordering[i - 1] is in cumulative_sets_rl[i], el needs to be picked if we do not
        # pick any of the preceding elements as there are no further elements to cover this branch.
        if ordering[i - 1] in cumulative_sets_rl[i]:
            required_right.add(el)
    # If we do it the other direction we can do the same thing for any following elements.
    cumulative_sets_lr = [None for _ in range(len(ordering))]
    cumulative_sets_lr[0] = current_set.intersection(parallel_sets[ordering[0]])
    required_left = {ordering[0]}
    for i in range(0, len(ordering) - 1):
        el = ordering[i + 1]
        cumulative_sets_lr[i + 1] = cumulative_sets_lr[i].intersection(parallel_sets[el])
        # similar reasoning - if we pick none of the elements after this one, there would be
        # a uncovered branch
        if ordering[i + 1] in cumulative_sets_lr[i]:
            required_left.add(el)
    # Elements that are in both required sets are always to be taken.
    always_required = required_left.intersection(required_right)

    # For future additions: - if one skips elements that have already been investigated previously (i.e., 
    # elsewhere in the ordering, another indicator is important to keep track of:
    # cumulative_sets_lr[-1] and cumulative_sets_rl[0] should always be empty sets - if they are not
    # there exists an element that is not optional that was excluded.
    # Probably shouldn't happen since we force always required, but just in case, handle this edge case.
    if len(cumulative_sets_lr[-1]) != 0 or len(cumulative_sets_rl[0]) != 0:
        if verbose: print("forbidden case - no choices cover all branches anymore...")
        return

    fixed_set = [partial(set_idx, i, 1)]
    fixed_restore = [partial(set_idx, i, 0)]
    
    if verbose: print(f"in this case to cover all branches {always_required} are required")
    for a in always_required:
        current_set.intersection_update(parallel_sets[a])
        fixed_set.append(partial(set_idx, a, 1))
        fixed_restore.append(partial(set_idx, a, 0))

    if len(current_set) == 0:
        if verbose: print(f"fixed case - no more free choices left to make after picking {i}")
        yield fixed_set, fixed_restore
    else:
        if verbose: print(f"recursive case for {i}")
        for e in current_set:
            # consider the cases where we pick it
            
            print(f"considering picking {e}")
            for (set_list, restore_list) in enumerate_parallel_set_recur(g, parallel_sets, e, current_set, unpickable=unpickable, ref_og=ref_og):
                yield (fixed_set + set_list), (fixed_restore + restore_list)
            print(f"no longer considering picking {e}")
            # now - for the following picks consider the case where we not allow e to be picked anymore.
            unpickable.add(i)
        # to avoid issues with branching allow picking these elements again if another branch investigates them.
        for e in current_set:
            unpickable.remove(i)