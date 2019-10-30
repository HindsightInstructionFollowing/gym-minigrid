from gym_minigrid.minigrid import MiniGridEnv, Grid, Ball, Key, Floor, Wall, Goal, Box, Door, Lava, \
    OBJECT_TO_IDX, COLOR_TO_IDX, SHADE_TO_IDX, IDX_TO_COLOR, IDX_TO_OBJECT, IDX_TO_SHADE, IDX_TO_SIZE, SIZE_TO_IDX

from enum import IntEnum

import numpy as np
from gym import spaces

class GridAttr(Grid):
    def __init__(self, width, height):
        """
        Remove channel "Opened or not" and add other attributes to an object
        """
        super().__init__(width=width, height=height)

    def encode(self, vis_mask=None):
        """
        Produce a compact numpy encoding of the grid
        """

        if vis_mask is None:
            vis_mask = np.ones((self.width, self.height), dtype=bool)

        array = np.zeros((self.width, self.height, 3), dtype='uint8')
        for i in range(self.width):
            for j in range(self.height):
                if vis_mask[i, j]:
                    v = self.get(i, j)

                    if v is None:
                        array[i, j, 0] = OBJECT_TO_IDX['empty']
                        array[i, j, 1] = 0
                        array[i, j, 2] = 0
                        array[i, j, 3] = 0
                        array[i, j, 4] = 0

                    else:
                        # State, 0: open, 1: closed, 2: locked
                        state = 0
                        if hasattr(v, 'is_open') and not v.is_open:
                            state = 1
                        if hasattr(v, 'is_locked') and v.is_locked:
                            state = 2

                        array[i, j, 0] = OBJECT_TO_IDX[v.type]
                        array[i, j, 1] = COLOR_TO_IDX[v.color]
                        array[i, j, 2] = SHADE_TO_IDX[v.shade]
                        array[i, j, 3] = SIZE_TO_IDX[v.size]
                        array[i, j, 4] = state
        return array

    @staticmethod
    def decode(array):
        """
        Decode an array grid encoding back into a grid
        """

        width, height, channels = array.shape
        assert channels == 3

        grid = Grid(width, height)
        for i in range(width):
            for j in range(height):
                typeIdx, colorIdx, shadeIdx, sizeIdx, state = array[i, j]

                if typeIdx == OBJECT_TO_IDX['unseen'] or \
                        typeIdx == OBJECT_TO_IDX['empty']:
                    continue

                objType = IDX_TO_OBJECT[typeIdx]
                color = IDX_TO_COLOR[colorIdx]
                shade = IDX_TO_SHADE[shadeIdx]
                size = IDX_TO_SIZE[size]

                # State, 0: open, 1: closed, 2: locked
                is_open = state == 0
                is_locked = state == 2

                if objType == 'wall':
                    v = Wall(color)
                elif objType == 'floor':
                    v = Floor(color)
                elif objType == 'ball':
                    v = Ball(color, shade=shade, size=size)
                elif objType == 'key':
                    v = Key(color, shade=shade, size=size)
                elif objType == 'box':
                    v = Box(color)
                elif objType == 'door':
                    v = Door(color, is_open, is_locked)
                elif objType == 'goal':
                    v = Goal()
                elif objType == 'lava':
                    v = Lava()
                else:
                    assert False, "unknown obj type in decode '%s'" % objType
                grid.set(i, j, v)

        return grid