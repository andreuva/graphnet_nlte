from read_header import read_generic_header
from read_nodes import get_data
from configs import PMD
import numpy as np
import os, sys
import copy

data = copy.deepcopy(PMD)
data['head'] = read_generic_header()
data['data'] = get_data(data)
