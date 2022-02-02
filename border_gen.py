# this script contains the code for the border generator object
import random
from bisect import bisect_left
from typing import List, Dict, Set


class BorderGenerator:

    freq_list: List[int]
    current_dict: Dict[int, int]
    st_weight: int
    yielded: Set[int]

    def __init__(self, border_dict: Dict[int, int], st_weight: int):
        freq_list = []
        for pixel, num_neighbors in border_dict.items():
            exp_value = num_neighbors ** st_weight
            for _ in range(exp_value):
                freq_list.append(pixel)

        self.freq_list = sorted(freq_list)
        self.current_dict = border_dict
        self.st_weight = st_weight
        self.yielded = set()

    def __next__(self):
        if len(self.freq_list) == len(self.yielded):
            raise Exception('No pixels are valid')

        while True:
            return_idx = random.randint(0, len(self.freq_list) - 1)
            if return_idx not in self.yielded:
                self.yielded.add(return_idx)
                break

        return self.freq_list[return_idx]

    def __iter__(self):
        return self

    def update(self, update_dict: Dict[int, int]) -> None:
        for pixel, num_neighbors in update_dict.items():
            neighbor_diff = num_neighbors - (current_entries := self.current_dict[pixel])
            if neighbor_diff > 0:
                to_add = ((current_entries + neighbor_diff) ** self.st_weight) - (current_entries ** self.st_weight)
                add_idx = bisect_left(self.freq_list, pixel)

                for _ in range(to_add):
                    self.freq_list.insert(add_idx, pixel)
            elif neighbor_diff < 0:
                to_remove = (current_entries ** self.st_weight) - ((current_entries + neighbor_diff) ** self.st_weight)
                remove_idx = bisect_left(self.freq_list, pixel)

                for _ in range(to_remove):
                    self.freq_list.pop(remove_idx)
            self.current_dict[pixel] = num_neighbors

        self.yielded = set()
