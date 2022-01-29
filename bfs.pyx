#cython: language_level=3
import numpy as np
cimport numpy as cnp
cnp.import_array()


def district_breaks(int swap_pixel, dict to_search, dict pix_num_to_row, cnp.ndarray[long, ndim=2] neighbors,
                    cnp.ndarray[long, ndim=1] districts):
    cdef list queue, neighbor_list
    cdef set visited
    cdef cnp.ndarray[long, ndim=1] focus_neighbors
    cdef int class_num
    cdef int start, focus, focus_neighbor

    for class_num, neighbor_list in to_search.items():
        start = neighbor_list.pop()
        visited = {start}
        queue = [start]

        while queue:
            focus = queue.pop(0)
            if focus == -1:
                continue
            if focus in neighbor_list:
                neighbor_list.remove(focus)
                if not neighbor_list:
                    break

            focus_neighbors = neighbors[pix_num_to_row[focus]]
            for idx in range(len(focus_neighbors)):
                focus_neighbor = focus_neighbors[idx]
                if focus_neighbor == -1:
                    continue
                if (
                    focus_neighbor not in visited and
                    districts[pix_num_to_row[focus_neighbor]] == class_num and
                    focus_neighbor != swap_pixel
                ):
                    visited.add(focus_neighbor)
                    queue.append(focus_neighbor)

        if neighbor_list:
            return True

    return False
