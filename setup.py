import numpy as np
import pandas as pd
import itertools as it
import os
import networkx as nx
import copy
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
#import seaborn as sns
#from ipywidgets import interact
import webbrowser
import IPython.display as ipd
from IPython.display import SVG, Image, HTML


#plt.style.use("fivethirtyeight")
#plt.rc("figure", figsize=(3,2))
#sns.set_palette('deep')
pd.options.display.show_dimensions = True

decimals = 3
def num_format(z):
    if np.abs(z) < 1e-16:
        z = 0
    z = np.real_if_close(z)    
    return np.round(z,decimals)

def display(X, rows=None, where="inline", filename="df"):
    if(rows == 'all'):
        rows = 2000
    elif(type(rows) is int):
        rows *= 2
    else:
        rows = 100

    if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series) or (isinstance(X, np.ndarray) and X.ndim <=2):        
        X = pd.DataFrame(X)
        num = X.select_dtypes(include=['number'])
        num = num.applymap(num_format)
        X[num.columns] = num        
        if(where == "popup"):
            filename = name + ".html"
            X.to_html(filename)        
            webbrowser.open(filename,new=2)
        else:
            pd.set_option('display.max_rows', rows)
            ipd.display(X)
            pd.reset_option('display.max_rows')
    else:
        ipd.display(X)