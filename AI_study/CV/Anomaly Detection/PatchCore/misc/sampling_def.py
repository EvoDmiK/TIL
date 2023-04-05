import abc

import numpy as np


class SamplingMethod(object):
    
    __metaclass__ = abc.ABCMeta
    
    @abc.abstractmethod
    def __init__(self, x, y, seed, **kwargs):
        
        self.x    = x
        self.y    = y
        self.seed = seed
        
        
    def flatten_x(self):
        
        shape  = self.x.shape
        flat_x = self.x
        
        if len(shape) > 2: flat_x = np.reshape(self.x, (shape[0], np.product(shape[1:])))
        
        return flat_x
    
    
    @abc.abstractmethod
    def _select_batch(self): return


    def select_batch(self, **kwargs): return self._select_batch(**kwargs) 


    def to_dict(self): return None