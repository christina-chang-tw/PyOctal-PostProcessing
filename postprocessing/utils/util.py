class DictObj:
    """ Convert a dictionary to python object """
    def __init__(self, **dictionary):
        for key, val in dictionary.items():
            if isinstance(val, dict):
                self.__dict__[key] = DictObj(**val)
            else:
                self.__dict__[key] = val

    def __getitem__(self, key):
        return self.__dict__[key]

    def keys(self):
        return self.__dict__.keys()

    def get(self, key):
        if key in self.__dict__.keys():
            return self.__getitem__(key)
        return None
    
    def items(self):
        return self.__dict__.items()