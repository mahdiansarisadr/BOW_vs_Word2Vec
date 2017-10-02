
from pandas import DataFrame
import pandas as pd
import matplotlib.pyplot as plt
import psycopg2
from pandas import Series
from collections import Counter
import numpy as np
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.externals import joblib
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.preprocessing import PolynomialFeatures, Imputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
import seaborn as sns
from xgboost import XGBClassifier
#%matplotlib inline

#import textblob as TextBlob
import matplotlib as mpl
import os
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.svm import SVC
import datetime
from sklearn.metrics import confusion_matrix
import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.naive_bayes import GaussianNB
import pickle
#import ms_qa_functions as ms
from sklearn.cross_validation import StratifiedKFold

import gzip
import  json
from pandas import DataFrame
import os
import pandas as pd
#from sqlalchemy import create_engine
#from sqlalchemy_utils import database_exists, create_database
from pandas import Series
import numpy as np
from bs4 import UnicodeDammit
import datetime
from collections import Counter
import sys

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn import metrics



def get_softskill_table(company_ans_list):
    company_ids = ['bba34b19a9ec6032acdc8c9b', '3505ddf363aca5cc5e436445',
       '8ae806508303922ee5ad28fc', 'ce7dcc15bc15e82f99672291',
       '4d469859040b84d869068942', 'da52905fe18b021d34bc170e',
       '87a33d0ecf39dac411b891fc', '90a3d65fff8bce5014a4fc4a',
       '838a40459e9738760e69a3bd', 'ee04ea55145d7c4f109f5186',
       'fceb435380c769e9d3c0750a', '4aaa2993debd5ff76ffc3e35',
       '8b4f08115e0311968d7857f2', '1c6874619fb84539cbb63128',
       '6ddf1d9d85cec0fbe4fbae2d', 'f7321223ab0b3a000addcec4',
       '424d1c4eb54081eb01fc5d55', '4c9f902fbf1b11a54b6f0fd9',
       'b162fe39b3239e930ea106ee', '3233d9bcb1c5292ecdf92665']

    company_question_dict ={
         0: 'Section 2: Softskills',
         1: 'Salutation - First Impressions Are The Most Important',
         2: 'Empathy',
         #3:  N/A,
         4: 'Personable Voice/Tone',
         5: 'Personalization',
         6: 'Tone',
         7: 'Greeting', #['Customer Engagement', 'Greeting'],
         8: 'Language',
         #9: N/A,
         10: 'Quality',
         11: 'Soft', #['Soft', 'Greetings'],
         12: 'Greeting',
         #13:
         #14:
         15: 'Style and tone',
         16: 'Empathy', #['Tone and Professionalism', 'Connection', 'Empathy', 'Greeting'],
         17: 'Customer Care',
         #18: N/A
         19: 'Style & Voice'
    }
    s_table_list = list()
    for id in [0,1,2,4,5,6,7,8,10,11,12,15,16,17,19]:
        company_ans_table = company_ans_list[id]
        question_name = company_question_dict[id]
        company_id = company_ids[id]
        s_table  = company_ans_table[['comp_gradable_id', question_name]]
        print(id)
        print(s_table.columns)
        s_table.dropna(inplace=True)
        s_table = s_table.rename(columns = {question_name: 'soft_score'})
        s_table_list.append(s_table)

    final_s_table = pd.concat(s_table_list)
    return  final_s_table


def question_list(table_answers, company_id):
    company_answers = table_answers[table_answers['composer_id']==company_id]
    qlist = list(set(company_answers['vectorizedSecNames'].sum()))
    return (company_answers, qlist)

def insert_scores(A, col_name):
    temp_dict = dict()
    for k, v in A['normalizedSectionScores'].iteritems():
        for key in v.keys():
            if v[key]['name'] == col_name:
                temp_dict[k] = v[key]['score']
    new_col = pd.Series(temp_dict , name = col_name)
    #A = pd.concat([new_col, A], axis=1)
    return new_col

def company_answers_list(table_answers):
    table_answers['comp_gradable_id'] = table_answers['composer_id'] + table_answers['externalGradableID'].astype('str')
    table_answers['vectorizedSecNames'] = table_answers['normalizedSectionScores'].apply(lambda x: [x[key]['name'] for key in x.keys()])


    compay_id_list  = table_answers['composer_id'].unique()
    company_ans_list = []
    for company_id in compay_id_list:
        (company_answers, qlist) = question_list(table_answers, company_id)
        for col in qlist:
            A = insert_scores(company_answers, col)
            company_answers = pd.concat([company_answers, A], axis=1)
        company_ans_list.append(company_answers)
    #company_ans_list.to_csv('company_ans_list.csv')
    return company_ans_list




def data_preparation(table_answers, table_gradable):
    company_ans_list = company_answers_list(table_answers)
    #company_ans_list = pd.read_csv('company_ans_list.csv')
    softskill_table = get_softskill_table(company_ans_list)

    softskill_text_scores = table_gradable[['comp_gradable_id', 'agent_words', 'agent_conversations']].merge(softskill_table, on='comp_gradable_id', how='inner')
    return softskill_text_scores
