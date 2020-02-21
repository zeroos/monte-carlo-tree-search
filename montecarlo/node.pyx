import cython
from libc.math cimport log, sqrt


@cython.freelist(1000)
cdef class Node:
    cdef readonly tuple state
    cdef public double win_value
    cdef readonly long visits
    cdef public Node parent
    cdef public object children
    cdef public char expanded
    cdef double discovery_factor

    def __init__(self, state):
        self.state = state
        self.win_value = 0
        self.visits = 0
        self.parent = None
        self.children = []
        self.expanded = False
        self.discovery_factor = 0.35


    def update_win_value(self, value):
        self.win_value += value
        self.visits += 1

        if self.parent:
            self.parent.update_win_value(value)

    def add_child(self, child):
        self.children.append(child)
        child.parent = self

    def add_children(self, children):
        for child in children:
            self.add_child(child)

    def get_preferred_child(self):
        cdef float score
        cdef float best_score = -1
        cdef Node child
        cdef Node best_child

        for child in self.children:
            score = child.get_score()

            if score > best_score:
                best_score = score
                best_child = child
        if best_score == -1:
            raise IndexError()
        return best_child

    cdef double get_score(self):
        cdef double parent_visits
        parent_visits = self.parent.visits
        discovery_operand = (
            self.discovery_factor *
            sqrt(log(parent_visits) / (self.visits or 0.00001))
        )

        win_operand = self.win_value / (self.visits or 1)

        return win_operand + discovery_operand

    def is_scorable(self):
        return True
