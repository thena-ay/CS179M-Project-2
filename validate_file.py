import pandas as pd
import os
import numpy as np

def validate_file(filename, sep = "   "):
    if not filename.endswith(".txt"):
        raise Exception("This file is not a .txt file. ABORT")
    
    if not os.path.exists(filename):
        raise Exception("This file does not exist. ABORT")
    
    df = pd.read_csv(filename, sep= sep, engine='python', header=None)
    
    if df.shape[1] == 3:
        df = df.drop(df.columns[0], axis=1)
    df = df.apply(pd.to_numeric, errors='coerce')
    df = df.to_numpy()
    return df