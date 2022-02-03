from typing import Tuple
import numpy as np


class Element:

    def __init__(
        self,
        shape=None,
        xy_coords_botleft=None,
        artist=None,
    ):
        self.shape = shape
        self.xy_coords_botleft = xy_coords_botleft
        self.artist = artist


    @property
    def xy_coords_mean(self):
        return self.xy_coords_botleft + self.shape / 2

    @property
    def xy_coords_topleft(self):
        return self.xy_coords_topleft + np.array([self.shape[0], 0])
    
    @property
    def xy_coords_topright(self):
        return self.xy_coords_topleft + self.shape
    
    @property
    def xy_coords_botright(self):
        return self.xy_coords_topleft + np.array([0, self.shape[0]])
    

    def set_xy_coords_mean(self, new_coords: Tuple):
        assert self.shape is not None, "Must give shape to give coords mean. Else give coords botleft and mean."
        new_coords = np.array(new_coords)
        self.xy_coords_botleft = new_coords - self.shape / 2

    def set_xy_coords_botleft(self, new_coords: Tuple):
        self.xy_coords_botleft = np.array(new_coords)



class ElementImage(Element):

    def __init__(self, image, *args, **kwargs):
        super().__init__(shape=np.array(image.shape), *args, **kwargs)
        self.image = image


    @property
    def img(self):
        return self.image
