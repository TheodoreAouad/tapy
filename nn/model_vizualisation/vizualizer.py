from multiprocessing.dummy import Array
from typing import Tuple, Union, Any

import matplotlib.pyplot as plt
import numpy as np
import cv2


class Element:

    def __init__(
        self,
        shape=None,
        xy_coords_botleft=None,
        xy_coords_mean=None,
    ):
        if xy_coords_botleft is not None and xy_coords_mean is not None:
            raise ValueError("choose either xy coords botleft or mean.")

        self.shape = shape
        self.xy_coords_botleft = None

        if xy_coords_botleft is not None:
            self.set_xy_coords_botleft(xy_coords_botleft)
        if xy_coords_mean is not None:
            self.set_xy_coords_mean(xy_coords_mean)

    def translate(self, vector: np.ndarray):
        self.set_xy_coords_botleft(self.xy_coords_botleft + vector)

    @property
    def xy_coords_mean(self):
        return self.xy_coords_botleft + self.shape / 2

    @property
    def barycentre(self):
        return self.xy_coords_mean

    @property
    def xy_coords_topleft(self):
        return self.xy_coords_topleft + np.array([self.shape[0], 0])

    @property
    def xy_coords_topright(self):
        return self.xy_coords_topleft + self.shape

    @property
    def xy_coords_botright(self):
        return self.xy_coords_topleft + np.array([0, self.shape[0]])

    @property
    def xy_coords_midright(self):
        return self.barycentre + np.array([self.shape[0] / 2, 0])

    @property
    def xy_coords_midleft(self):
        return self.barycentre - np.array([self.shape[0] / 2, 0])

    @property
    def xy_coords_midtop(self):
        return self.barycentre + np.array([0, self.shape[1] / 2])

    @property
    def xy_coords_midbot(self):
        return self.barycentre - np.array([0, self.shape[1] / 2])

    @property
    def xy_coords_midbottom(self):
        return self.xy_coords_midbot

    def is_inside_element(self, coords):
        coords = np.array(coords)
        return (self.xy_coords_botleft <= coords <= self.xy_coords_topright).all()

    def set_xy_coords_mean(self, new_coords: Tuple):
        assert self.shape is not None, "Must give shape to give coords mean. Else give coords botleft and mean."
        new_coords = np.array(new_coords)
        self.xy_coords_botleft = new_coords - self.shape / 2

    def set_xy_coords_botleft(self, new_coords: Tuple):
        self.xy_coords_botleft = np.array(new_coords)

    def add_to_canva(self, canva: "Canva"):
        raise NotImplementedError


class ElementImage(Element):

    def __init__(self, image: np.ndarray, new_shape=None, imshow_kwargs={}, *args, **kwargs):
        super().__init__(shape=np.array(image.shape), *args, **kwargs)
        self.image = image
        self.imshow_kwargs = imshow_kwargs

        if new_shape is not None:
            self.resize(new_shape)


    @property
    def img(self):
        return self.image

    def add_to_canva(self, canva: "Canva", new_shape=None, coords=None, coords_type="barycentre", imshow_kwargs=None):
        if new_shape is not None:
            self.resize(new_shape)

        if imshow_kwargs is None:
            imshow_kwargs = self.imshow_kwargs

        if coords is not None:
            if coords_type == "barycentre":
                self.set_xy_coords_mean(coords)
            elif coords_type == "botleft":
                self.set_xy_coords_botleft(coords)

        canva.ax.imshow(self.image, extent=(
            self.xy_coords_botleft[0], self.xy_coords_botleft[0] + self.shape[0],
            self.xy_coords_botleft[1], self.xy_coords_botleft[1] + self.shape[1],
        ), **imshow_kwargs)


    def resize(self, new_shape: Union[Union[float, int], Tuple[int]], interpolation=cv2.INTER_AREA):
        if isinstance(new_shape, (float, int)):
            new_shape = [int(new_shape), int(new_shape)]

        self.image = cv2.resize(self.image, (new_shape[1], new_shape[0]), interpolation=interpolation)
        self.shape = np.array(new_shape)
        return self.image


class ElementArrow(Element):

    def __init__(
        self,
        x, y, dx, dy, arrow_kwargs=dict(),
        **kwargs
    ):
        super().__init__(
            shape=np.array([np.abs(dx), np.abs(dy)]),
            xy_coords_mean=np.array([x, y]) + .5 * np.array([dx, dy]),
            **kwargs
        )
        self.x, self.y, self.dx, self.dy = x, y, dx, dy
        self.arrow_kwargs = arrow_kwargs

    def add_to_canva(self, canva: "Canva", ):
        return canva.ax.arrow(self.x, self.y, self.dx, self.dy, **self.arrow_kwargs)

    @staticmethod
    def link_elements(
        elt1: Element,
        elt2: Element,
        link1="adapt",
        link2="adapt",
        length_includes_head=True, width=.1, **kwargs
    ):
        if link1 == "adapt" or link2 == "adapt":
            link1, link2 = ElementArrow.adapt_link(elt1, elt2, link1, link2)

        if isinstance(link1, str):
            x1, y1 = getattr(elt1, f"xy_coords_{link1}")
        elif isinstance(link1, (tuple, np.ndarray)):
            x1, y1 = link1
        else:
            raise ValueError('link1 must be string or tuple or numpy array.')

        if isinstance(link2, str):
            x2, y2 = getattr(elt2, f"xy_coords_{link2}")
        elif isinstance(link2, (tuple, np.ndarray)):
            x2, y2 = link2
        else:
            raise ValueError('link2 must be string or tuple or numpy array.')

        kwargs.update({
            "length_includes_head": length_includes_head,
            "width": width
        })

        return ElementArrow(x1, y1, x2-x1, y2-y1, arrow_kwargs=kwargs)

    @staticmethod
    def adapt_link(elt1, elt2, link1, link2):
        dx, dy = elt2.barycentre - elt1.barycentre

        if link1 == "adapt":
            if dx > 0:
                link1 = "midright"
            elif dx < 0:
                link1 = "midleft"
            elif dy > 0:
                link1 = "midtop"
            else:
                link1 = "midbot"

        if link2 == "adapt":
            if dx < 0:
                link2 = "midright"
            elif dx > 0:
                link2 = "midleft"
            elif dy < 0:
                link2 = "midtop"
            else:
                link2 = "midbot"

        return link1, link2


class ElementGrouper(Element):

    def __init__(self, elements=dict(), *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.elements = dict()
        for key, element in elements.items():
            self.add_element(element, key=key)
        self.xy_coords_botleft = np.array([0, 0])  # TODO: adapt this to the elements
        self.shape = np.array([0, 0])  # TODO: adapt this to the elements

    def translate_group(self, vector: np.ndarray):
        # self.translate(vector)
        for element in self.elements.values():
            element.translate(vector)

    def add_element(self, element: Element, key=None):
        if key is None:
            key = 0
            while key in self.elements.keys():
                key = np.random.randint(0, 9999999)

        self.elements[key] = element

    def add_to_canva(self, canva: "Canva"):
        for key, element in self.elements.items():
            if element in canva.elements.values():
                continue

            if key in canva.elements.keys():
                idx = 0
                while f'{key}_{idx}' in canva.elements.keys():
                    idx += 1
                canva.add_element(element, key=f'{key}_{idx}')
            else:
                canva.add_element(element, key=key)

            canva.add_element(element, key)

    def __len__(self):
        return len(self.elements)



class Canva:

    def __init__(
        self,
        elements=dict(),
        xlim=None,
        ylim=None,
        lim_mode='adaptable',
        **kwargs
    ):
        self.fig, self.ax = plt.subplots(1, 1, **kwargs)
        self.ax.axis('off')
        if xlim is None:
            xlim = self.ax.get_xlim()
        if ylim is None:
            ylim = self.ax.get_ylim()

        self.elements = elements
        self.lim_mode = lim_mode
        self.xmin, self.xmax, self.ymin, self.ymax = None, None, None, None

        self.set_xlim(xlim)
        self.set_ylim(ylim)


        if self.lim_mode == 'adaptable':
            self.ax.set_xlim(auto=True)
            self.ax.set_ylim(auto=True)

    @property
    def xlim(self):
        return self.xmin, self.xmax

    @property
    def ylim(self):
        return self.ymin, self.ymax

    def set_xlim(self, left: float = None, right: float = None):
        if isinstance(left, tuple):
            left, right = left

        if left is not None:
            self.xmin = left
            self.ax.set_xlim(left=left)

        if right is not None:
            self.xmax = right
            self.ax.set_xlim(right=right)

    def set_ylim(self, bottom: float = None, top: float = None):
        if isinstance(bottom, tuple):
            bottom, top = bottom

        if bottom is not None:
            self.ymin = bottom
            self.ax.set_ylim(bottom=bottom)

        if top is not None:
            self.ymax = top
            self.ax.set_ylim(top=top)

    def set_lims(self, xlim: Tuple[float], ylim: Tuple[float]):
        self.set_xlim(xlim)
        self.set_ylim(ylim)
        return xlim, ylim

    def add_element(self, element: Element, key: Any = None, *args, **kwargs):
        if key is None:
            key = 0
            while key in self.elements.keys():
                key = np.random.randint(0, 9999999)

        self.elements[key] = element
        element.add_to_canva(self, *args, **kwargs)

        if self.lim_mode == "adaptable":
            self.adapt_lims_to_element(element.shape, element.xy_coords_botleft)

        return self

    def adapt_lims_to_element(self, shape: Tuple[float, float], xy_coords_botleft: Tuple[float, float]):
        new_xmin = xy_coords_botleft[0]
        if self.xmin > new_xmin:
            self.set_xlim(left=new_xmin)

        new_xmax = new_xmin + shape[0]
        if self.xmax < new_xmax:
            self.set_xlim(right=new_xmax)

        new_ymin = xy_coords_botleft[1]
        if self.ymin > new_ymin:
            self.set_ylim(bottom=new_ymin)

        new_ymax = new_ymin + shape[1]
        if self.ymax < new_ymax:
            self.set_ylim(top=new_ymax)

    def show(self):
        self.fig.show()
        return self.fig
