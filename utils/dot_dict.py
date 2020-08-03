class DotDict(dict):
    """
    A class to convert a nested Dictionary into an object with key-values
    that are accessible using attribute notation (DotDict.attribute) instead of
    key notation (Dict["key"]). This class recursively sets Dicts to objects,
    allowing you to recurse down nested dicts (like: DotDict.attr.attr)
    
    From kadee
    https://stackoverflow.com/questions/4984647/accessing-dict-keys-like-an-attribute
    """
    def __init__(self, iterable, **kwargs):
        super(DotDict, self).__init__(iterable, **kwargs)
        for key, value in iterable.items():
            if isinstance(value, dict):
                self.__dict__[key] = DotDict(value)
            else:
                self.__dict__[key] = value
                