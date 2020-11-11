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

    def flatten(self, d=None):
        """Flatten (possibly) nested dictionaries to all be `AttrDict`'s."""
        if d is None:
            d = self.__dict__

        if not isinstance(d, AttrDict):
            d = AttrDict(**d)

        for key, val in d.items():
            if isinstance(val, dict):
                if not isinstance(val, AttrDict):
                    d[key] = self.flatten(val)
        #
        #
        #  for key, val in d.items():
        #      if isinstance(val, dict):
        #          if not isinstance(val, AttrDict):
        #              return self.flatten(val)
        #          d[key] = AttrDict(val)
        #  return d
        return d
