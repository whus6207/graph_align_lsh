import numpy as np 
import pandas as pd
import os.path
import pickle

fname = 'exp_runtime.pkl'
if os.path.isfile(fname):
  with open(fname, 'rb') as f:
    df = pickle.load(f)
    df.to_csv(fname + '.csv')