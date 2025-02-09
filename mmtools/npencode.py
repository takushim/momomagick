#!/usr/bin/env python
# implemented referring to https://github.com/hmallen/numpyencoder

import json
import numpy as np

class NumpyEncoder (json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)

        elif isinstance(obj, np.floating):
            return float(obj)
        
        elif isinstance(obj, np.complexfloating):
            return {'real': obj.real, 'imag': obj.imag}
        
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
    
        elif isinstance(obj, (bool, np.bool_)):
            return bool(obj)

        elif isinstance(obj, np.void): 
            return None

        return json.JSONEncoder.default(self, obj)
