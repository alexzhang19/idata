#!/usr/bin/env python3
# coding: utf-8

import random
from .compose import *
from idata.utils.type import is_dict

__all__ = ["Lambda", "RandomTransforms", "RandomApply", "RandomOrder", "RandomChoice"]


class Lambda(object):
    """Apply a user-defined lambda as a transform.
    Args:
        lambd (function): Lambda/function to be used for transform.
    """

    def __init__(self, lambd):
        assert callable(lambd), repr(type(lambd).__name__) + " object is not callable"
        self.lambd = lambd

    def __call__(self, data):
        return self.lambd(data)

    def __repr__(self):
        return self.__class__.__name__ + '()'


# random transforms
class RandomTransforms(object):
    """Base class for a list of transformations with randomness
    Args:
        transforms (list or tuple): list of transformations
    """

    def __init__(self, transforms):
        assert isinstance(transforms, (list, tuple))
        self.transforms = transforms

    def __call__(self, *args, **kwargs):
        raise NotImplementedError()

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class RandomApply(RandomTransforms):
    """Apply randomly a list of transformations with a given probability
    Args:
        transforms (list or tuple): list of transformations
        p (float): probability
    """

    def __init__(self, transforms, p=0.5):
        super(RandomApply, self).__init__(transforms)

        if type(p) is float or type(p) is int:
            p = [p for _ in range(len(transforms))]

        assert len(p) == len(transforms)
        self.p = p

    def __call__(self, *imgs):
        for i, t in enumerate(self.transforms):
            a = random.random()
            if self.p[i] < a:
                continue
            if type(imgs) is tuple:
                imgs = t(*imgs)
            else:
                imgs = t(imgs)
        return imgs[0] if len(imgs) is 1 else imgs

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += '\n    p={}'.format(self.p)
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class RandomOrder(RandomTransforms):
    """Apply a list of transformations in a random order
    """

    def __call__(self, img):
        order = list(range(len(self.transforms)))
        random.shuffle(order)
        for i in order:
            img = self.transforms[i](img)
        return img


class RandomChoice(RandomTransforms):
    """Apply single transformation randomly picked from a list
    """

    def __call__(self, img):
        t = random.choice(self.transforms)
        return t(img)


if __name__ == "__main__":
    import torch as t
    import numpy as np


    def target_to_tensor(target):
        target = np.array(target)
        return t.from_numpy(target).type(t.LongTensor)


    target_transform = Lambda(target_to_tensor)
