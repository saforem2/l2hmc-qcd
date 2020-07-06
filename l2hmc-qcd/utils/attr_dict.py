"""
attr_dict.py

Implements AttrDict class for defining an attribute dictionary that allows us
to address keys as if they were attributes.

Idea taken from:
    https://danijar.com/patterns-for-fast-prototyping-with-tensorflow/

Author: Sam Foreman
Date: 04/09/2019
"""


class AttrDict(dict):
    """A dict which is accessible via attribute dot notation."""
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self
