#  DAEDALUS – Distributed and Automated Evolutionary Deep Architecture Learning with Unprecedented Scalability
# 
# This research code was developed as part of the research programme Open Technology Programme with project number 18373, which was financed by the Dutch Research Council (NWO), Elekta, and Ortec Logiqcare.
# 
# Project leaders: Peter A.N. Bosman, Tanja Alderliesten
# Researchers: Alex Chebykin, Arthur Guijt, Vangelis Kostoulas
# Main code developer: Arthur Guijt

from typing import List, Union, Iterable, Tuple, Any
import numpy as np

class KeyedPriorityQueue:

    def __init__(self, initial_items: Iterable[Tuple[int, Any]] = []):
        self.items = [(priority, key, value) for key, (priority, value) in enumerate(initial_items)]
        self.keys_locs = np.arange(len(self.items))
        self.next_key = len(self.items)
        self.reuse_keys: List[int] = []

        # Heapify
        for i in range(len(self.items) - 1, 0 - 1, -1):
            self._siftDown(i)
            
    def is_empty(self) -> bool:
        return len(self.items) == 0

    def add(self, priority, value) -> int:
        key = self._get_unused_key()
        index = len(self.items)
        self.keys_locs[key] = index
        self.items.append((priority, key, value))
        self._siftUp(index)
        return key

    def update(self, key, new_priority):
        index = self.keys_locs[key]
        orig_item = self.items[index]
        orig_priority = orig_item[0]
        self.items[index] = (new_priority, orig_item[1], orig_item[2])

        if orig_priority > new_priority:
            # Priority has gone up - apply siftup.
            self._siftUp(index)
        else:
            # Priority has gone down, apply siftdown
            self._siftDown(index)

    def remove(self, key):
        index = self.keys_locs[key]
        self.keys_locs[key] = -1

        removed = self.items[index]
        replacement = self.items.pop()

        if index < len(self.items):
            # if the position still exists, we need to use the replacement
            self.items[index] = replacement
            self.keys_locs[replacement[1]] = index

            if replacement[0] > removed[0]:
                self._siftDown(index)
            else:
                self._siftUp(index)
        else:
            # if the position no longer exists, we just removed ourselves
            # and no re-heaping is necessary.
            pass
        
        # make key available for reuse
        self.reuse_keys.append(key)

    def popmin(self):
        popped = self.items[0]
        self.remove(popped[1])
        return popped

    def _left_of(self, index):
        return index * 2 + 1

    def _right_of(self, index):
        return index * 2 + 2
    
    def _parent_of(self, index):
        return (index - 1) >> 1

    
    def _get_unused_key(self):
        if len(self.reuse_keys) == 0:
            key = self.next_key
            self.keys_locs = np.concatenate([self.keys_locs, [-1]])
            self.next_key += 1
        else:
            key = self.reuse_keys.pop()
            self.keys_locs[key] = -1
        return key

    def _siftDown(self, index):
        while True:
            left_index = self._left_of(index)
            right_index = self._right_of(index)

            # If we are a leaf, we can stop
            if left_index >= len(self.items): return
            
            if right_index < len(self.items):
                # If both children are valid, we need to find the one with the smallest priority of the two.
                min_child_priority, min_child_index = min((self.items[left_index][0], left_index), (self.items[right_index][0], right_index))
            else:
                # If we only have one child (only happens for one node in the heap, if the heap is odd-sized)
                # This is always the leftmost child.
                min_child_priority, min_child_index = self.items[left_index][0], left_index

            # Can stop sifting down if we are smaller (or equal) to the smallest of them.
            current_priority = self.items[index][0]
            if min_child_priority >= current_priority: return

            # If we are not, swap with the smallest of the two, ensuring the heap property is met, here. To do so:
            # - swap key locations
            key_parent = self.items[index][1]
            key_child = self.items[min_child_index][1]
            self.keys_locs[[key_parent, key_child]] = self.keys_locs[[key_child, key_parent]]
            # - swap items
            self.items[index], self.items[min_child_index] = self.items[min_child_index], self.items[index]
            # Finally, update index for next iteration
            index = min_child_index

    def _siftUp(self, index):
        while True:
            parent_index = self._parent_of(index)
            # If we are at the root, we can stop.
            if parent_index < 0: return

            current_priority = self.items[index][0]
            parent_priority = self.items[parent_index][0]

            # If the ordering is right (parent <= current), we can stop
            if parent_priority <= current_priority: return

            # Swap keys
            key_parent = self.items[parent_index][1]
            key_child = self.items[index][1]
            self.keys_locs[[key_parent, key_child]] = self.keys_locs[[key_child, key_parent]]
            # Swap items
            self.items[index], self.items[parent_index] = self.items[parent_index], self.items[index]

            index = parent_index