#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 14:24:31 2017

@author: keriabermudez
"""


import gzip
import  json
from pandas import DataFrame
import os
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy_utils import database_exists, create_database
from pandas import Series
import numpy as np
from bs4 import UnicodeDammit
import datetime
from collections import Counter
          
import ms_qa_functions as ms
import sys

df = np.random.normal(10,10)
ms.extract_policy(df, 1, 'a' )
