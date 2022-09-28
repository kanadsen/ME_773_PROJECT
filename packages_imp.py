'''
Packages to be imported in the entire program
Installations required for packages 
'''

import tensorflow as tf
import numpy as np
import pandas as pd
import sklearn
import os
import seaborn as sns
from sklearn.model_selection import train_test_split
import sys
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import random
import json
import papermill as pm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm  #tqdm is used for progress bar
from io import StringIO 
import argparse
import requests
from sklearn.metrics import confusion_matrix, accuracy_score  #Scikit Learn