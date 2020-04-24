import ast
import six
import copy
import json


def eval_str_fn(val):
    if val.lower() in {"true", "false"}:
        return val.lower() == "true"

    try:
        return ast.literal_eval(val)
    except ValueError:
        return val


class Config(object):
    def __init__(self, config_dict=None):
        self.update(config_dict)

    def __setattr__(self, k, v):
        self.__dict__[k] = Config(v) if isinstance(v, dict) else copy.deepcopy(v)
    
    def __getattr__(self, k):
        return self.__dict__[k]
    
    def __repr__(self):
        return repr(self.as_dict())
    
    def __str__(self):
        try:
            return json.dumps(self.as_dict(), indent=4)
        except TypeError:
            return str(self.as_dict())
    
    def _update(self, config_dict, allow_new_keys=True):
        """Recursively update internal members."""
        if not config_dict:
            return 
        
        for k, v in six.iteritems(config_dict):
            if k not in self.__dict__.keys():
                if allow_new_keys:
                    self.__setattr__(k, v)
                else:
                    raise KeyError("Key `{}` does not exist for overriding.".format(k))
            else:
                if isinstance(v, dict):
                    self.__dict__[k]._update(v, allow_new_keys)
                else:
                    self.__dict__[k] = copy.deepcopy(v)
    
    def get(self, k, default_value=None):
        return self.__dict__.get(k, default_value)
    
    def update(self, config_dict):
        """Update members while allowing new keys."""
        self._update(config_dict, allow_new_keys=True)
    
    def override(self, config_dict_or_str):
        """Update members while disallowing new keys."""
        if isinstance(config_dict_or_str, str):
            config_dict = self.parse_from_str(config_dict_or_str)
        elif isinstance(config_dict_or_str, dict):
            config_dict = config_dict_or_str
        else:
            raise ValueError("Unkown value type: {}".format(config_dict_or_str))
        
        self._update(config_dict, allow_new_keys=False)
    
    def parse_from_str(self, config_str):
        """parse from a string in format 'x=a,y=2' and return the dict."""
        if not config_str:
            return {}
        
        config_dict = {}
        try:
            for kv_pair in config_str.split(","):
                if not kv_pair:   # skip empty string
                    continue
                k, v = kv_pair.split("=")
                config_dict[k.strip()] = eval_str_fn(config_str)
            
            return config_dict
        except ValueError:
            raise ValueError("Invalid config_str: {}.".format(config_str))
    
    def as_dict(self):
        """Returns a dict representation."""
        config_dict = {}
        for k, v in six.iteritems(self.__dict__):
            if isinstance(v, Config):
                config_dict[k] = v.as_dict()
            else:
                config_dict[k] = copy.deepcopy(v)
        
        return config_dict

