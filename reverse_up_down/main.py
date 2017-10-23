import numpy as np
import tensorflow as tf
from tre_simple import *
from parser import *
from E_M_utils import *


epoche = 100

data_set = dataset_parser()

pi,sp_p,A,bi = training(data_set,epoche)
