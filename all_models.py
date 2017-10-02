
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
color = sns.color_palette()


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
from imblearn.under_sampling import RandomUnderSampler
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


def plot_precision_recall(y_test, probas, graph_name):
    path_graphs = '/Users/mehdiansari/Desktop/MaestroQA/Data/Figures/'
    ap = average_precision_score(y_test, probas[:, 1])
    precision, recall, thresholds = precision_recall_curve( y_test,  probas[:, 1], pos_label=1)
    fig, ax1 = plt.subplots(nrows=1,ncols=1, figsize = (6,6))
    ax1.plot(recall,precision,lw=1,label='area = %0.2f'% ap)
    ax1.set_xlim([-0.05, 1.05])
    ax1.set_ylim([-0.05, 1.05])
    ax1.set_xlabel('recall')
    ax1.set_ylabel('precision')
    ax1.set_title('Precision-Recall Curve')
    ax1.legend(loc="lower right")
    fig.savefig(path_graphs+graph_name+'_precision_recall.png')
    plt.show()

def BOW_LR(table_gradable,  model_name, author='agent'):

    table_gradable['qa_score_below_90']   =  (table_gradable['qa_score']< 90).astype('float')
    table_gradable = table_gradable[table_gradable.n_back_forth > 1]

    if author == 'agent':
        bow_table = table_gradable[['agent_words', 'qa_score_below_90']]
        bow_table.dropna(inplace=True)

    if author == 'user':
        bow_table = table_gradable[['user_words', 'qa_score_below_90']]
        bow_table.dropna(inplace=True)

    agent_words = list(bow_table['agent_words'].values)
    label = list(bow_table['qa_score_below_90'].values)
    X_train, X_test, y_train, y_test = train_test_split( agent_words, label, test_size=0.3, random_state=0)


    if model_name == 'LR':
        text_clf = Pipeline([('vect', CountVectorizer()),
                          ('tfidf', TfidfTransformer(use_idf=True)),
                          ('clf', LogisticRegression(random_state=0,class_weight= 'balanced' )),])

    if model_name =='RF':
        text_clf = Pipeline([('vect', CountVectorizer()),
                          ('tfidf', TfidfTransformer(use_idf=True)),
                          ('clf', RandomForestClassifier(random_state=0,class_weight= 'balanced',
                                                          n_estimators = 100, n_jobs=-1 )),])
    if model_name =='XGB':


        text_clf = Pipeline([('vect', CountVectorizer()),
                      ('tfidf', TfidfTransformer()),
                      ('clf', XGBClassifier(scale_pos_weight=4)),])



    text_clf.fit(X_train, y_train)
    y_pred = text_clf.predict(X_test)
    probas = text_clf.predict_proba(X_test)
    print(y_pred)

    feature_imp =text_clf.named_steps['clf'].feature_importances_
    word_mapper = text_clf.named_steps['vect'].vocabulary_

    graph_name = author+'BOW'+model_name
    #plot_precision_recall(y_test, probas, graph_name)

    return (y_test, probas, feature_imp, word_mapper)


def prepare_X_y_sentiment(table_gradable):
    columns = [
     #'csat_score',
     'ag_n_sentences',
     'us_n_sentences',
     'ag_n_words',
     'us_n_words',
     'ag_grateful_feature',
     'us_grateful_feature',
     'ag_assistance_feature',
     'ag_apologetic_feature',
     'us_confusion_feature',
     'us_frustration_feature',
     'ag_senti_polarity_mean',
     'ag_senti_polarity_min',
     'us_senti_polarity_mean',
     'us_senti_polarity_min',
     'time_to_resolve',
     'time_to_response',
     'n_back_forth',
     'ag_policy_feature',
     'ag_senti_polarity_first','ag_senti_polarity_last',
     'us_senti_polarity_first', 'us_senti_polarity_last']

    table_gradable['qa_score_below_90']   =  (table_gradable['qa_score']< 90).astype('float')
    table_gradable = table_gradable[table_gradable.n_back_forth > 1]

    X_d = table_gradable.ix[:,columns].copy()
    X_data = pd.DataFrame(Imputer(strategy = 'median').fit_transform(X_d), columns = X_d.columns)
    index_vals = X_data.index


    y_data = table_gradable['qa_score_below_90'].copy()
    y_data  = y_data.values

    return X_data, y_data

def bow_score(X_train, X_test, y_train, author_words, max_ngram, min_ngram):
    message  = 'calculating bow score for'+author_words + '...'
    print(message)
    model_name_bow ='LR'
    X_train_bow = list(X_train[author_words].copy().values)
    X_test_bow = list(X_test[author_words].copy().values)
    y_train_bow = list(y_train.copy().values)
    #y_test_bow = list(y_test.copy().values)

    if model_name_bow == 'LR':
        text_clf = Pipeline([('vect', CountVectorizer(ngram_range=(min_ngram, max_ngram))),
                      ('tfidf', TfidfTransformer()),
                      ('clf', LogisticRegression(random_state=0,class_weight= 'balanced' )),])
        text_clf.fit(X_train_bow, y_train_bow)



    if model_name_bow =='RF':
        score_param = 'roc_auc'
        param_range = [10,20,50,100]
        param_grid = [
                      {
                        'clf__bootstrap': [True, False],
                        'clf__n_estimators':param_range,
                       'clf__criterion':['gini','entropy'],
                       }
                      ]

        pip_rf = Pipeline([('vect', CountVectorizer(ngram_range=(min_ngram, max_ngram))),
                      ('tfidf', TfidfTransformer()),
                      ('clf', RandomForestClassifier(random_state=0,class_weight= 'balanced' )),])
        cv = StratifiedKFold(y=y_train, n_folds=8)
        pip_gr = GridSearchCV(pip_rf, param_grid,
                       scoring=score_param, #recall
                       cv=cv,
                       verbose=1,
                       n_jobs=-1)
        pip_gr.fit(X_train_bow, y_train_bow)
        text_clf = pip_gr.best_estimator_


    if model_name_bow =='XGB':
        score_param = 'roc_auc'
        param_range = [30,50,100]
        param_grid = [
                      {
                        #'clf__bootstrap': [True, False],
                        'clf__n_estimators':param_range,
                       #'clf__criterion':['gini','entropy'],
                       'clf__scale_pos_weight': [2,3,4]
                         }
                      ]

        pip_xgb = Pipeline([('vect', CountVectorizer(ngram_range=(min_ngram, max_ngram))),
                      ('tfidf', TfidfTransformer()),
                      ('clf', XGBClassifier()),])

        cv = StratifiedKFold(y=y_train, n_folds=4)
        pip_gr = GridSearchCV(pip_xgb, param_grid,
                       scoring=score_param, #recall
                       cv=cv,
                       verbose=1,
                       n_jobs=-1)
        pip_gr.fit(X_train_bow, y_train_bow)
        text_clf = pip_gr.best_estimator_

    if model_name_bow =='NB':
        text_clf = Pipeline([('vect', CountVectorizer(ngram_range=(min_ngram, max_ngram))),
                      ('tfidf', TfidfTransformer()),
                      ('clf', MultinomialNB()),])
        text_clf.fit(X_train_bow, y_train_bow)



    bow_train_score = text_clf.predict_proba(X_train_bow)
    bow_test_score = text_clf.predict_proba(X_test_bow)
    return (bow_train_score, bow_test_score)



def combined_BOW_sentiment(table_gradable, max_ngram):
    columns = [
     #'csat_score',
     'ag_n_sentences',
     'us_n_sentences',
     'ag_n_words',
     'us_n_words',
     'ag_grateful_feature',
     'us_grateful_feature',
     'ag_assistance_feature',
     'ag_apologetic_feature',
     'us_confusion_feature',
     'us_frustration_feature',
     'ag_senti_polarity_mean',
     'ag_senti_polarity_min',
     'us_senti_polarity_mean',
     'us_senti_polarity_min',
     'time_to_resolve',
     'time_to_response',
     'n_back_forth',
     'ag_policy_feature',
     'ag_senti_polarity_first','ag_senti_polarity_last',
     'us_senti_polarity_first', 'us_senti_polarity_last']


    table_gradable['qa_score_below_90']   =  (table_gradable['qa_score']< 90).astype('float')
    table_gradable = table_gradable[table_gradable.n_back_forth > 1]

    if max_ngram>2:
        table_gradable.dropna(subset = [['agent_words', 'agent_conversations']], inplace = True)
        X_d = table_gradable.ix[:,columns + ['agent_words', 'agent_conversations']].copy()
    else:
        table_gradable.dropna(subset = ['agent_words'], inplace = True)
        X_d = table_gradable.ix[:,columns + ['agent_words']].copy()

    #X_data = pd.DataFrame(preprocessing.Imputer(strategy = 'median').fit_transform(X_d), columns = X_d.columns)
    #X_data_combined = X_data.merge(table_gradable[['comp_gradable_id', 'agent_words']], on = 'comp_gradable_id')

    X_data_combined = X_d.dropna()
    y_data = table_gradable['qa_score_below_90'].copy()

    ########## first split all of the data
    X_train, X_test, y_train, y_test = train_test_split(X_data_combined, y_data, test_size=0.3, random_state=0)




    ################################### combine  features of bag of words and sentiment
    ################################### combine  features of bag of words and sentiment

    X_train_ensemble = X_train.copy()
    X_test_ensemble = X_test.copy()

    ############################ data prepration and model for bag of user words
    (agent_bow_train_score, agent_bow_test_score) = bow_score(X_train, X_test, y_train, 'agent_words', max_ngram,1)
    X_train_ensemble['agent_words'] = agent_bow_train_score[:,1]
    X_test_ensemble['agent_words'] = agent_bow_test_score[:,1]

    if max_ngram>2:
        min_ngram = 3
        min_ngram = min(min_ngram, max_ngram)
        (ag_conv_bow_train_score, ag_conv_bow_test_score)   = bow_score(X_train, X_test, y_train, 'agent_conversations', max_ngram, min_ngram)
        X_train_ensemble['agent_conversations'] = ag_conv_bow_train_score[:,1]
        X_test_ensemble['agent_conversations'] = ag_conv_bow_test_score[:,1]



    ###########################################################################

    ################ Ensemble model
    ensemble_model ='LR'
    print('training the ensemble model ...')
    ############################################################################# Logistic regression
    if ensemble_model == 'Simple_LR':
            score_param = 'roc_auc'
            param_range = [1.0 ]
            param_grid = [
                          {
                           'clf__penalty':['l2'],
                           'clf__C': param_range}
                          ]
            pip_lgr = Pipeline([('scl', StandardScaler()),
                               # not working  ('pca', PCA()),
                                 ('clf', LogisticRegression(random_state=0,class_weight= 'balanced' ))])#class_weight= 'balanced'
            #cv = StratifiedKFold(y=y_train, n_folds=3)
            #gs_lgr = GridSearchCV(pip_lgr, param_grid,
            #                       scoring=score_param,
            #                       cv = cv,# cv =10 StratifiedKFold().fit(X_train, y_train)
            #                       verbose=1,
            #                           n_jobs=-1)
            #gs_lgr.fit(X_train_ensemble.values, y_train.values)
            #clf = gs_lgr.best_estimator_
            clf = pip_lgr.fit(X_train_ensemble.values, y_train.values)
    ############################################################################# Logistic regression
    if ensemble_model == 'LR':
            score_param = 'roc_auc'
            param_range = [0.01,0.1,1.0,10.0,100.0 ]
            param_grid = [
                          {
                           'clf__penalty':['l1','l2'],
                           'clf__C': param_range}
                          ]
            pip_lgr = Pipeline([('scl', StandardScaler()),
                               # not working  ('pca', PCA()),
                                 ('clf', LogisticRegression(random_state=0,class_weight= 'balanced' ))])#class_weight= 'balanced'
            cv = StratifiedKFold(y=y_train, n_folds=5)
            gs_lgr = GridSearchCV(pip_lgr, param_grid,
                                   scoring=score_param,
                                   cv = cv,# cv =10 StratifiedKFold().fit(X_train, y_train)
                                   verbose=1,
                                   n_jobs=-1)

            gs_lgr.fit(X_train_ensemble.values, y_train.values)
            clf = gs_lgr.best_estimator_

    ############################################################################# Random forest
    if ensemble_model == 'RF':
            score_param = 'roc_auc'
            param_range = [20,50,100]
            param_grid = [
                          {
                            'clf__bootstrap': [True, False],
                            'clf__n_estimators':param_range,
                           'clf__criterion':['gini','entropy'],
                           }
                          ]

            pip_rf = Pipeline([('scl', StandardScaler()),
                                 ('clf', RandomForestClassifier(random_state=0,class_weight= 'balanced' ))])
            cv = StratifiedKFold(y=y_train, n_folds=6)
            pip_rf = GridSearchCV(pip_rf, param_grid,
                                       scoring=score_param, #recall
                                       cv=cv,
                                       verbose=1,
                                       n_jobs=-1)

            pip_rf.fit(X_train_ensemble.values, y_train.values)
            #print("Pipeline Paremeters: \n")
            #print (pip_rf.get_params)
            clf = pip_rf.best_estimator_

    ##############################################

    y_pred = clf.predict(X_test_ensemble.values)
    probas = clf.predict_proba(X_test_ensemble.values)
    graph_name = 'ensemble'
    #plot_precision_recall(y_test, probas, graph_name)
    return (y_test, probas)




def sentiment_model(table_gradable, model):
    X_data, y_data = prepare_X_y_sentiment(table_gradable)
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.3, random_state=0)


    score_param = 'roc_auc'
    if model == 'Simple_LR':

            pip_lgr = Pipeline([('scl', StandardScaler()),('clf', LogisticRegression(random_state=0,class_weight= 'balanced' ))])
            clf = pip_lgr.fit(X_train, y_train)


    if model == 'LR':
            score_param = 'roc_auc'
            param_range = [0.1,1.0,10.0 ]
            param_grid = [
                          {
                           'clf__penalty':['l2'],
                           'clf__C': param_range}
                          ]
            pip_lgr = Pipeline([('scl', StandardScaler()),
                               # not working  ('pca', PCA()),
                                 ('clf', LogisticRegression(random_state=0,class_weight= 'balanced' ))])#class_weight= 'balanced'
            cv = StratifiedKFold(y=y_train, n_folds=5)
            gs_lgr = GridSearchCV(pip_lgr, param_grid,
                                   scoring=score_param,
                                   cv = cv,# cv =10 StratifiedKFold().fit(X_train, y_train)
                                   verbose=1,
                                   n_jobs=-1)

            gs_lgr.fit(X_train, y_train)
            clf = gs_lgr.best_estimator_

    ############################################################################# Random forest
    if model == 'RF':
        param_range = [20,50,100]
        param_grid = [
                      {
                        'clf__bootstrap': [True],
                        'clf__n_estimators':param_range,
                       'clf__criterion':['gini'],
                       }
                      ]

        pip_rf = Pipeline([('scl', StandardScaler()),
                             ('clf', RandomForestClassifier(random_state=0,class_weight= 'balanced' ))])
        cv = StratifiedKFold(y=y_train, n_folds=6)
        pip_rf = GridSearchCV(pip_rf, param_grid,
                                   scoring=score_param, #recall
                                   cv=cv,
                                   verbose=1,
                                   n_jobs=-1)

        pip_rf.fit(X_train, y_train)
        print("Pipeline Paremeters: \n")
        print (pip_rf.get_params)
        clf = pip_rf.best_estimator_




    y_pred = clf.predict(X_test)
    probas = clf.predict_proba(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    print('accuracy_score:', accuracy_score(y_test, y_pred))
    train_acc = accuracy_score(y_train, clf.predict(X_train))
    graph_name = 'sentiment_RF'
    #plot_precision_recall(y_test, probas, graph_name)
    return (y_test, probas)#(1-train_acc,1- test_acc)



def multiplot_precision_recall(table_gradable, max_ngram):
    path_graphs = '/Users/mehdiansari/Desktop/MaestroQA/Data/Figures/'
    fig, ax1 = plt.subplots(nrows=1,ncols=1, figsize = (6,6))
    ax1.set_xlim([-0.05, 1.05])
    ax1.set_ylim([-0.05, 1.05])
    ax1.set_xlabel('recall')
    ax1.set_ylabel('precision')
    ax1.set_title('Precision-Recall Curve')
    #ax1.axhline(y=bad_tickets.iloc[0], color = 'k',linestyle =':',lw=2)

    for n_gram in range(max_ngram):
        (y_test, probas) = combined_BOW_sentiment(table_gradable, n_gram+1)
        ap = average_precision_score(y_test, probas[:, 1])
        precision, recall, thresholds = precision_recall_curve( y_test,  probas[:, 1], pos_label=n_gram+1)
        ax1.plot(recall,precision,lw=1,label='area = %0.2f'% ap)
    ax1.legend(loc="lower right")
    graph_name = 'compare_all'
    fig.savefig(path_graphs+graph_name+'_precision_recall.png')
    plt.show()


def multiplot_precision_recall_seaborn(table_gradable, max_ngram):
    print('Hey')
    path_graphs = '/Users/mehdiansari/Desktop/MaestroQA/Data/Figures/'
    (y_test, probas) = combined_BOW_sentiment(table_gradable, 1)
    ap = average_precision_score(y_test, probas[:, 1])
    precision, recall, thresholds = precision_recall_curve( y_test,  probas[:, 1], pos_label=1)

    plt.figure(figsize=(12,8))
    sns.regplot(x=recall, y=precision)
    (y_test, probas) = combined_BOW_sentiment(table_gradable, 3)
    ap = average_precision_score(y_test, probas[:, 1])
    precision, recall, thresholds = precision_recall_curve( y_test,  probas[:, 1], pos_label=1)
    sns.regplot(x=recall, y=precision)

    plt.ylabel('Log error', fontsize=12)
    plt.xlabel('Bathroom Count', fontsize=12)
    plt.show()


def compare_precision_recall_plot(table_gradable):


    (y_test,probas) = sentiment_model(table_gradable, 'LR')
    ap1 = average_precision_score(y_test, probas[:, 1])

    precision1, recall1, thresholds = precision_recall_curve( y_test,  probas[:, 1], pos_label=1)
    fig, ax1 = plt.subplots(nrows=1,ncols=1, figsize = (6,4))
    ax1.set_xlim([-0.05, 1.05])
    ax1.set_ylim([-0.05, 1.05])
    ax1.set_xlabel('recall')
    ax1.set_ylabel('precision')
    ax1.set_title('Precision-Recall Curve')
    #ax1.axhline(y=bad_tickets.iloc[0], color = 'k',linestyle =':',lw=2)

    #(y_test,probas, x, y) = BOW_LR(table_gradable, 'LR')
    (y_test,probas) = combined_BOW_sentiment(table_gradable, 1)

    ap2 = average_precision_score(y_test, probas[:, 1])

    precision2, recall2, thresholds = precision_recall_curve(y_test,  probas[:, 1], pos_label=1)
    #fig, ax1 = plt.subplots(nrows=1,ncols=1, figsize = (6,4))
    ax1.plot(recall1,precision1,lw=1,label='Sentiment')#: (AUC = %0.2f)'% ap1)
    ax1.plot(recall2,precision2,lw=1,label='Combined')#: (AUC = %0.2f)'% ap2)
    ax1.axhline(y=.215, color = 'k',linestyle =':',lw=2)
    ax1.axvline(x=.2, color = 'b',linestyle =':',lw=2)

    ax1.legend(loc="upper right")
    plt.show()
    path_graphs = '/Users/mehdiansari/Desktop/MaestroQA/Data/Figures/'
    fig.savefig(path_graphs+'BOW_vs_sentiment_precision_recall.png')


def softskill_model(table_gradable,  model_name):

    table_gradable.dropna(inplace=True)
    table_gradable['qa_score_below_90']   =  (table_gradable['soft_score']< 90).astype('float')
    #table_gradable = table_gradable[table_gradable.n_back_forth > 1]


    agent_words = list(table_gradable['agent_words'].values)
    label = list(table_gradable['qa_score_below_90'].values)
    X_train, X_test, y_train, y_test = train_test_split( agent_words, label, test_size=0.3, random_state=0)


    if model_name == 'LR':
        text_clf = Pipeline([('vect', CountVectorizer()),
                          ('tfidf', TfidfTransformer()),
                          ('clf', LogisticRegression(random_state=0,class_weight= 'balanced' )),])

    if model_name =='RF':
        text_clf = Pipeline([('vect', CountVectorizer()),
                          ('tfidf', TfidfTransformer()),
                          ('clf', RandomForestClassifier(random_state=0,class_weight= 'balanced',
                                                          n_estimators = 100, n_jobs=-1 )),])


    text_clf.fit(X_train, y_train)
    y_pred = text_clf.predict(X_test)
    probas = text_clf.predict_proba(X_test)
    print(y_pred)
    graph_name = 'SoftSkill_BOW_'+model_name
    plot_precision_recall(y_test, probas, graph_name)

    return text_clf



def plot_word_importace(table_gradable):
    table_gradable['agent_words'] = table_gradable['agent_words'].apply(lambda x: x.replace('dir', ''))
    table_gradable['agent_words'] = table_gradable['agent_words'].apply(lambda x: x.replace('com', ''))
    table_gradable['agent_words'] = table_gradable['agent_words'].apply(lambda x: x.replace('www', ''))

    (y_test, probas, feature_imp, word_mapper) = am.BOW_LR(table_gradable.sample(frac=.2), 'RF')
    sorted_dictionary = sorted(word_mapper.items())
    sorted_list = [x[0] for x in sorted_dictionary]
    mapped = {y:x for x,y in zip(feature_imp, sorted_list)}
    inverse_mapped = {y:x for x,y in mapped.items()}
    sorted_words = sorted(inverse_mapped.items(), reverse= True)
    df = pd.DataFrame.from_records(sorted_words, columns=['imp', 'words'])

    table_gradable = table_gradable.dropna(subset=['agent_words'])

    bads = table_gradable[table_gradable['qa_score']<90]
    goods = table_gradable[table_gradable['qa_score']>=90]
    first_most = 19
    df['sign'] = np.nan
    sign_list = list()
    for word in df.words[:first_most]:
       # if not(word == 'dir' or word=='com' or word=='www'):
            df_bad  = bads['agent_words'].apply(lambda x: word in x)
            r_bad = df_bad.sum()/df_bad.shape[0]
            df_good  = goods['agent_words'].apply(lambda x: word in x)
            r_good = df_good.sum()/df_good.shape[0]
            sign_list.append(np.sign(r_bad - r_good))
            #print(r_bad - r_good)
            #print(np.sign(r_bad - r_good))


    df['sign'][:first_most] = sign_list
    import matplotlib.patches as mpatches
    fig, ax = plt.subplots(nrows=1,ncols=1, figsize = (6,4))
    clrs = ['blue' if (x > 0) else 'red' for x in df['sign'][:first_most]]
    ax = sns.barplot(x = 'imp', y = 'words',
                     data = df[:first_most],
                     palette=clrs)
    ax.set(xlabel='Feature importance', ylabel='Words' )

    red_patch = mpatches.Patch(color='red', label='bad')
    blue_patch = mpatches.Patch(color='blue', label='good')
    ax.legend(handles=[red_patch, blue_patch], loc="lower right")

    path_graphs = '/Users/mehdiansari/Desktop/MaestroQA/Data/Figures/'
    fig.tight_layout()
    fig.savefig(path_graphs+'Colored_Feature_imp_words.png')
    plt.show()

def roc_curve_plot_sent_vs_BOW(table_gradable,path_graphs,model_name):
    #######ROC Curve input y_test and probas, path graphs and model name
    (y_test,probas) = sentiment_model(table_gradable, 'LR')
    fpr, tpr, thresholds = roc_curve(y_test,probas[:, 1], pos_label=1)
    roc_auc = auc(fpr, tpr)


    ###################################
    fig, ax1 = plt.subplots(nrows=1,ncols=1, figsize = (6,4))
    ax1.plot(fpr,tpr,lw=1,label='Sentiment: (AUC = %0.2f)'% roc_auc)

    #(y_test,probas, x, y) = BOW_LR(table_gradable, 'LR')
    (y_test,probas) = combined_BOW_sentiment(table_gradable, 1)
    fpr, tpr, thresholds = roc_curve(y_test,probas[:, 1], pos_label=1)
    roc_auc = auc(fpr, tpr)
    ax1.plot(fpr,tpr,lw=1,label='Combined: (AUC = %0.2f)'% roc_auc)



    ax1.plot([0, 0, 1],
             [0, 1, 1],
             lw=2,
             linestyle=':',
             color='black',
             label='perfect performance')

    ax1.set_xlim([-0.05, 1.05])
    ax1.set_ylim([-0.05, 1.05])
    ax1.set_xlabel('false positive rate')
    ax1.set_ylabel('true positive rate')
    ax1.set_title('Receiver Operator Characteristic')
    ax1.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path_graphs+model_name+'_ROC.png')
    plt.show()
