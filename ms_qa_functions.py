#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 21:05:31 2017

@author: keriabermudez
"""

import gzip
import  json
from pandas import DataFrame
import os
import re
import pandas as pd
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
from sqlalchemy_utils import database_exists, create_database
import psycopg2
from pandas import Series
from collections import Counter
import numpy as np
from textblob import TextBlob
from nltk.corpus import stopwords # Import the stop word list
import nltk
from nltk import WordNetLemmatizer
from bs4 import UnicodeDammit
from textblob import Word
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import confusion_matrix
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve


#%%
def process_gradable(row):
    if row.get('tags') == None:
        row['tags'] = ['empty']
    else:
        tags = row['tags'][1:-1]
        tags = tags.replace('"', '')
        tags = tags.replace(' ', '')
        row['tags_string'] = tags
    return row

#%%
def clean_tabs(body):
    converted = UnicodeDammit(body).unicode_markup
    converted = converted.replace("\n\n", " ")
    converted = converted.replace("\\n", " ")
    converted = converted.replace("\n", " ")
    converted = converted.replace("\\r", " ")
    converted = converted.replace("\r\r", " ")
    converted = converted.replace("\r\r", " ")
    converted = converted.replace("<br>", " ")
    converted = converted.replace("</a>", " ")
    converted = converted.replace("<a href=", " ")
    converted_cleaned = converted.replace("\r", " ")
    return converted_cleaned

def n_back_forth(table_gradable,table_comments):
    """
    Number of back and forth interactions
    """
    table_gradable['n_back_forth'] = 0
    for company in np.unique(table_gradable.composer_id):
        table_g = table_gradable[table_gradable.composer_id == company]
        table_c = table_comments[table_comments.composer_id == company]

        number_back_and_forth = table_c.comp_gradable_id.value_counts()
        # Checking number of back and forth ## Check thi
        table_g['n_back_forth'] = number_back_and_forth.copy()
        table_gradable[table_gradable.composer_id == company] = table_g
    return table_gradable

def clean_url_email(body):

    # eliminate  web addresses
    body = re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', ' ', body)
    #eliminate emails
    clean_body = re.sub(r'[\w\.-]+@[\w\.-]+',' ' , body)
    # eliminate digits
    #clean_body = re.sub("[^a-zA-Z]", " ", body)
    return clean_body

def preprocessing_words(body):

    if   isinstance(body, str):
        body = clean_url_email(body)

        body = re.sub("[^a-zA-Z]", " ", body)

         #eliminate proper nouns
        tagged_sentence = nltk.tag.pos_tag(body.split())
        edited_sentence = [word for word,tag in tagged_sentence if tag != 'NNP' and tag != 'NNPS']

        body = (' '.join(edited_sentence))


        # tokenize into words
        tokens = [word for sent in nltk.sent_tokenize(body) \
        for word in nltk.word_tokenize(sent)]

        # remove stopwords
        stop = stopwords.words('english')
        tokens = [token for token in tokens if token not in stop]

        # remove words less than three letters
        tokens = [word for word in tokens if len(word) >= 3]

        # lower capitalization
        tokens = [word.lower() for word in tokens]

        # lemmatize
        lmtzr = WordNetLemmatizer()
        tokens = [lmtzr.lemmatize(word) for word in tokens]
        tokens = [lmtzr.lemmatize(word,'v') for word in tokens]


        preprocessed_text= ' '.join(tokens)
    else:
        preprocessed_text = np.NaN
    return preprocessed_text

def clean_text(body):
    body = re.sub("[^a-zA-Z]", " ", body)
    #eliminate proper nouns
    tagged_sentence = nltk.tag.pos_tag(body.split())
    edited_sentence = [word for word,tag in tagged_sentence if tag != 'NNP' and tag != 'NNPS']

    body = (' '.join(edited_sentence))

    body = correct_spelling(body)
    # 3. Convert to lower case, split into individual words
    words = body.lower().split()
   # 4. In Python, searching a set is much faster than searching
    #   a list, so convert the stop words to a set
    stops = set(stopwords.words("english"))
    #
    # 5. Remove stop words
    meaningful_words = [w for w in words if not w in stops]
    #
    # 6. Join the words back into one string separated by space,
    # and return the result.
    clean_body = " ".join( meaningful_words )
    return clean_body

#%%
def count_grateful(body):
    body_blob = TextBlob(body)
    words = ['thank']
    thank_you_count = 0
    for word in words:
        thank_you_count+= body_blob.word_counts[word]
    return {'thank_you_count':thank_you_count}

def count_apologetic(body):
    body_blob = TextBlob(body)
    words = ['apologize',
             'sorry',
             'apology']
    apologize_count = 0
    for word in words:
        apologize_count+= body_blob.word_counts[word]
    return {'apologize_count':apologize_count}

def count_assistance(body):
    body_blob = TextBlob(body)
    words =['assistance',
            'assist',
            'offer',
            'extend',
            'suggest',
            'recommend',
            'propose'
            'help']
    assistance_count = 0
    for word in words:
        assistance_count+= body_blob.word_counts[word]
    return {'assistance_count':assistance_count}

def count_words(body):
    words = body.split()
    len(words)
    return {'n_words':float(len(words))}




def extract_policy(table_comments,column, author_type):
    table_comments= table_comments[table_comments.author_type == author_type]
    policy_feature = table_comments[column].apply(count_policy)
    policy_feature =  policy_feature.apply(pd.Series)
    policy_feature['comp_gradable_id'] = table_comments['comp_gradable_id'].copy()
    policy_feature = policy_feature.groupby('comp_gradable_id').sum()

    return policy_feature



def count_policy(body):
    body_blob = TextBlob(body)
    words = ['policy',
             'refund',
             'credit',
              'month',
               'day']
    policy_count = 0
    for word in words:
        policy_count+= body_blob.word_counts[word]
    return {'policy_count':policy_count}




def count_confusion(body):
    body_blob = TextBlob(body)
    words =['confusion',
            'confuse',
            'unclear',
            'lose',
            'mess',
            'understand',
             'puzzle']
    confusion_count = 0
    for word in words:
        confusion_count+= body_blob.word_counts[word]
    return {'confusion_count':confusion_count}

def count_frustration(body):
    body_blob = TextBlob(body)
    words =['anger',
            'frustration',
             'frustrate',
             'annoyance',
             'annoy',
             'disappointment',
             'disappoint',
             'sad',
             'dissatisfaction',
             'regret',
             'irritate',
             'discontent',

             ]
    frustration_count = 0
    for word in words:
        frustration_count+= body_blob.word_counts[word]
    return {'frustration_count':frustration_count}


def extract_len(body):
    len_text = len(body)
    body_blob = TextBlob(body)
    sentences = body_blob.sentences
    n_sentences = len(sentences)
    return {'len_text':len_text,'n_sentences':n_sentences}


def correct_spelling(body):
    body_blob = TextBlob(body)
    body_blob = body_blob.correct()
    #body_blob.word_counts['thank']
    return body_blob
"""
count misspells

"""
def count_spell_err(body):
    body = re.sub("[^a-zA-Z]", " ", body)
    text = TextBlob(body)
    count_misspells = 0

    for word in text.words:
        w = Word(word)
        fixed_w = w.spellcheck()
        if fixed_w[0][0] != word and fixed_w[0][1] > 0.95:
            count_misspells +=1
    return count_misspells

def count_spell_err_all(table_comments, column, author_type):
    table_comments = table_comments[table_comments.author_type == author_type]
    table_comments['spelling_feature'] = table_comments[column].apply(count_spell_err)
    grouped= table_comments.groupby(by ='comp_gradable_id').sum()
    spelling_feature = grouped['spelling_feature']
    return spelling_feature

'''
def time_to_res(table_comments, gd_id):
    us_comment = table_comments[(table_comments.comp_gradable_id == gd_id) & (table_comments.author_type == 'user')]
    us_comment_time = pd.to_datetime(us_comment.time)
    us_comment_time.sort_values(inplace = True)

    ag_comment = table_comments[(table_comments.comp_gradable_id == gd_id) & (table_comments.author_type == 'agent')]
    ag_comment_time = pd.to_datetime(ag_comment.time)
    ag_comment_time.sort_values(inplace = True)

    if (len(us_comment_time) <= 1) or (len(ag_comment_time) <= 1 ):
        time_to_resp = 0
        time_to_resol = 0

    else :
        min_time  = us_comment_time.iloc[0]
        time_to_resp = ag_comment_time.iloc[1] - min_time
        time_to_resp = time_to_resp.total_seconds()
        max_time = ag_comment_time.max()
        time_to_resol = max_time -min_time
        time_to_resol = time_to_resol.total_seconds()
    return time_to_resol, time_to_resp
'''

def time_to_res(table_comments):

    table_comments.sort_values(by='time', inplace= True)
    us_first_comment = table_comments[table_comments.author_type == 'user'].groupby('comp_gradable_id')['time'].first()
    ag_first_comment = table_comments[table_comments.author_type == 'agent'].groupby('comp_gradable_id')['time'].first()
    last_comment = table_comments.groupby('comp_gradable_id')['time'].last()


    us_first_comment = pd.to_datetime(us_first_comment)
    ag_first_comment = pd.to_datetime(ag_first_comment)
    last_comment = pd.to_datetime(last_comment)


    time_to_resol = last_comment - us_first_comment
    time_to_resp = ag_first_comment - us_first_comment



    time_to_resol =time_to_resol.dt.total_seconds()
    time_to_resp = time_to_resp.dt.total_seconds()



    time_to_resol.fillna(value = 0, inplace= True)
    time_to_resp.fillna(value = 0, inplace= True)

    time_to_resol[time_to_resol<0]=0
    time_to_resp[time_to_resp<0]=0
    time_to_resol.name = 'time_to_resolve'
    time_to_resp.name = 'time_to_response'

    return (time_to_resol, time_to_resp)

def extract_len_all(table_comments, column, author_type):
    table_comments= table_comments[table_comments.author_type == author_type]
    len_feature = table_comments[column].apply(extract_len)
    len_feature =  len_feature.apply(pd.Series)
    len_feature['comp_gradable_id'] = table_comments['comp_gradable_id'].copy()
    len_feature['author_type'] = table_comments['author_type'].copy()
    len_feature = len_feature.groupby('comp_gradable_id').sum()

    return len_feature

def extract_senti(body):
    body_blob = TextBlob(body)
    sentiment_polarity = body_blob.sentiment.polarity
    sentiment_subjectivity = body_blob.sentiment.subjectivity
    #body_blob.word_counts['thank']

    return {'senti_polarity':sentiment_polarity,'senti_subjectivity':sentiment_subjectivity}
# divide this
def extract_senti_ag_us(table_comments,column,author_type ):
    table_comments= table_comments[table_comments.author_type == author_type]
    senti_feature = table_comments['body'].apply(extract_senti)
    senti_feature =  senti_feature.apply(pd.Series)
    senti_feature['comp_gradable_id'] = table_comments['comp_gradable_id'].copy()
    senti_feature_mean = senti_feature.groupby('comp_gradable_id').mean()
    senti_feature_min = senti_feature.groupby('comp_gradable_id').min()

    return (senti_feature_mean,senti_feature_min)

def extract_n_words(table_comments,column, author_type):
    table_comments= table_comments[table_comments.author_type == author_type]
    total = table_comments[column].apply(count_words)
    total =  total.apply(pd.Series)
    total['comp_gradable_id'] = table_comments['comp_gradable_id'].copy()
    total = total.groupby('comp_gradable_id').sum()

    return total

def extract_grateful(table_comments,column, author_type):
    table_comments= table_comments[table_comments.author_type == author_type]
    grateful_feature = table_comments[column].apply(count_grateful)
    grateful_feature =  grateful_feature.apply(pd.Series)
    grateful_feature['comp_gradable_id'] = table_comments['comp_gradable_id'].copy()
    grateful_feature = grateful_feature.groupby('comp_gradable_id').sum()

    return grateful_feature

def extract_confusion(table_comments,column, author_type):
    table_comments= table_comments[table_comments.author_type == author_type]
    confusion_feature = table_comments[column].apply(count_confusion)
    confusion_feature =  confusion_feature.apply(pd.Series)
    confusion_feature['comp_gradable_id'] = table_comments['comp_gradable_id'].copy()
    confusion_feature = confusion_feature.groupby('comp_gradable_id').sum()

    return confusion_feature

def extract_frustration(table_comments,column, author_type):
    table_comments= table_comments[table_comments.author_type == author_type]
    frustration_feature = table_comments[column].apply(count_frustration)
    frustration_feature =  frustration_feature.apply(pd.Series)
    frustration_feature['comp_gradable_id'] = table_comments['comp_gradable_id'].copy()
    frustration_feature = frustration_feature.groupby('comp_gradable_id').sum()

    return frustration_feature

def extract_apologetic(table_comments,column,author_type):
    table_comments= table_comments[table_comments.author_type == author_type]

    apologetic_feature= table_comments[column].apply(count_apologetic)
    apologetic_feature =  apologetic_feature.apply(pd.Series)
    apologetic_feature['comp_gradable_id'] = table_comments['comp_gradable_id'].copy()
    apologetic_feature = apologetic_feature.groupby('comp_gradable_id').sum()

    return apologetic_feature

def extract_assistance(table_comments,column,author_type):
    table_comments= table_comments[table_comments.author_type == author_type]

    assistance_feature = table_comments[column].apply(count_assistance)
    assistance_feature = assistance_feature.apply(pd.Series)
    assistance_feature['comp_gradable_id'] = table_comments['comp_gradable_id'].copy()
    assistance_feature = assistance_feature.groupby('comp_gradable_id').sum()

    return assistance_feature


def roc_curve_plot(y_test,probas,path_graphs,model_name):
    #######ROC Curve input y_test and probas, path graphs and model name
    fpr, tpr, thresholds = roc_curve(y_test,
                                         probas[:, 1],
                                         pos_label=1)
    roc_auc = auc(fpr, tpr)
    print('The ROC auc is: %f ' % (roc_auc))
    fig, ax1 = plt.subplots(nrows=1,ncols=1, figsize = (6,4))
    ax1.plot(fpr,tpr,lw=1,label='ROC(area = %0.2f)'% roc_auc)
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
    

#%%
def confusion_matrix_plot(y_test,y_pred,model_name, path_graphs):
    #confusion matrix input y_test and ypred
    confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)
    print((confmat))
    fig, ax = plt.subplots(figsize=(2.5, 2.5))
    ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(confmat.shape[0]):
        for j in range(confmat.shape[1]):
            ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')

    plt.xlabel('predicted label')
    plt.ylabel('true label')
    plt.tight_layout()
    fig.savefig(path_graphs +model_name+'Confusion_Matrix.png')
    fig.savefig(path_graphs +model_name+'Confusion_Matrix.pdf')

def graph_predict_results(table_gradable,index_test,y_pred,y_test,probas,path_graphs, model_name):
    ## Graphs Input  table_gradable y test and y pred
    results_prediction = {'predicted_label':y_pred,'true_label':y_test,'probas_of_0': probas[:,0], 'probas_of_1' : probas[:,1]}
    results_prediction = DataFrame(results_prediction, index = index_test)

    #%%
    companies_testing = table_gradable.ix[index_test,'composer_id']
    results_prediction['qa_score'] = table_gradable.ix[index_test,'qa_score']

    results_prediction['companies'] = companies_testing
    results_prediction['false_positive'] = (results_prediction.predicted_label == 1) & (results_prediction.true_label == 0)
    results_prediction['true_positive'] = (results_prediction.predicted_label == 1) & (results_prediction.true_label == 1)
    grouped = results_prediction.groupby(by = 'companies').sum()

    #fp_tp = grouped.ix[:,['false_positive','true_positive']]
    total = grouped.ix[:,'false_positive']+ grouped.ix[:,'true_positive']
    grouped['fp_proportion'] = grouped['false_positive']/total
    grouped['tp_proportion'] = grouped['true_positive']/total

    # True positive and False Postive
    to_plot = grouped.ix[:,['tp_proportion','fp_proportion']]
    fig, ax1 = plt.subplots(nrows=1,ncols=1, figsize = (6,4))
    to_plot.plot.bar(stacked = True, ax = ax1)
    fig.savefig(path_graphs +model_name+'_true_pos_vs_neg.png')
    fig.savefig(path_graphs +model_name+'_true_pos_vs_neg.pdf')

    #%%
    # Total tickets to Grade per company
    n_total_10_model = 10/to_plot['tp_proportion']
    group_bad = table_gradable.groupby(by = 'composer_id' ).sum()
    group_total = table_gradable.groupby(by = 'composer_id' ).count()

    bad_tickets = group_bad['qa_score_below_90']/group_total['qa_score_below_90']
    good_tickets =  1- bad_tickets
    bad_tickets.rename('bad_tickets',inplace= True)
    good_tickets.rename('good_tickets',inplace= True)

    all_tickets = pd.concat([bad_tickets,good_tickets], axis =1)
    fig, ax1 = plt.subplots(nrows=1,ncols=1, figsize = (6,4))
    all_tickets.plot.bar(stacked = True, ax = ax1)
    fig.savefig(path_graphs+model_name+'_bad_vs_good.png')
    fig.savefig(path_graphs+model_name+'_bad_vs_good.pdf')

     # Total tickets to Grade per company
    n_total_10_random = 10/all_tickets['bad_tickets']
    total_10 = pd.concat([n_total_10_random,n_total_10_model], axis =1)
    total_10.rename(columns = {'bad_tickets':'random','tp_proportion':'model'}, inplace = True)
    fig, ax1 = plt.subplots(nrows=1,ncols=1, figsize = (6,4))
    total_10.plot.bar( ax = ax1)
    fig.savefig(path_graphs+model_name+'total_tickets_model_vs_rand.png')
    fig.savefig(path_graphs+model_name+'total_tickets_model_vs_rand.pdf')
    return bad_tickets, results_prediction

def precision_recall_plot(bad_tickets, y_test,probas,model_name, path_graphs):

    ####################  Precision and Recall Plot ####### Input y_test and probas
    ap = average_precision_score(y_test, probas[:, 1])
    precision, recall, thresholds = precision_recall_curve( y_test,  probas[:, 1], pos_label=1)
    fig, ax1 = plt.subplots(nrows=1,ncols=1, figsize = (6,4))
    ax1.plot(recall,precision,lw=1,label='area = %0.2f'% ap)
    ax1.set_xlim([-0.05, 1.05])
    ax1.set_ylim([-0.05, 1.05])
    ax1.set_xlabel('recall')
    ax1.set_ylabel('precision')
    ax1.set_title('Precision Recall')
    ax1.legend(loc="lower right")
    ax1.axhline(y=bad_tickets.iloc[0], color = 'k',linestyle =':',lw=2)

    fig.savefig(path_graphs+model_name+'precision_recall.pdf')
    fig.savefig(path_graphs+model_name+'precision_recall.png')


def number_tickets_plot(bad_tickets,precision_t,model_name, path_graphs):
    random = [bad_tickets,1-bad_tickets]
    model = [precision_t, 1-precision_t]

    fig3, ax1 = plt.subplots(nrows=1,ncols=1, figsize = (5,4))
    table = DataFrame([random,model],index =['random','model'], columns = ['bad tickets', 'good tickets'])
    table.plot.bar(stacked = True, ax = ax1)
    ax1.legend(loc='best')
    ax1.set_ylabel('Proportion Tickets')
    fig3.savefig(path_graphs+model_name+"proportion_optimized.pdf")


    fig4, ax2 = plt.subplots(nrows=1,ncols=1, figsize = (5,4))
    random_tickets = 10/bad_tickets
    model_tickets = 10/ precision_t
    ax2.bar([0,1],[random_tickets,model_tickets])
    ax2.bar([0,1],[10,10])
    ax2.set_xticks([0,1])
    ax2.set_xticklabels(['random','model'])
    ax2.set_ylabel('Number Tickets')
    fig4.savefig(path_graphs+model_name+"number_optimized.pdf")


def precision_rec_plot_target(bad_tickets, y_test,probas,precision_t,recall_t,model_name, path_graphs):

    ####################  Precision and Recall Plot ####### Input y_test and probas
    ap = average_precision_score(y_test, probas[:, 1])
    precision, recall, thresholds = precision_recall_curve( y_test,  probas[:, 1], pos_label=1)
    fig, ax1 = plt.subplots(nrows=1,ncols=1, figsize = (6,4))
    ax1.plot(recall,precision,lw=1,label='area = %0.2f'% ap)
    ax1.set_xlim([-0.05, 1.05])
    ax1.set_ylim([-0.05, 1.05])
    ax1.set_xlabel('recall')
    ax1.set_ylabel('precision')
    ax1.set_title('Precision Recall')
    ax1.legend(loc="lower right")
    ax1.axhline(y=bad_tickets, color = 'k',linestyle =':',lw=2)
    ax1.axhline(y=precision_t, color = 'k',linestyle =':',lw=2)
    ax1.axvline(x=recall_t, color = 'k',linestyle =':',lw=2)

    fig.savefig(path_graphs+model_name+'precision_recall.pdf')
    fig.savefig(path_graphs+model_name+'precision_recall.png')
    return fig

def get_threshold(table_prec_recall,bad_tickets,min_recall = 0.1):
        #min_recall = 0.2
    table_prec_recall = table_prec_recall[table_prec_recall.recall >min_recall]
    table_prec_recall[table_prec_recall.precision ==table_prec_recall.precision.max()]
    values =     table_prec_recall[table_prec_recall.precision ==table_prec_recall.precision.max()]
    return values


def summarize_company(company,table_gradable, table_comments):
    #process company
    table_g = table_gradable[table_gradable.composer_id == company]

    # Start Tags
    complete_list_tags = []
    for index in table_g.index:
        tags = table_g.ix[index,'tags']
        complete_list_tags +=  tags
    #End Tags

    #Score types
    complete_list_scoretypes = []
    for index in table_g.index:
        score_types = table_g.ix[index,'score_types']
        complete_list_scoretypes +=  score_types

    series_scores = Series(complete_list_scoretypes)

    c = Counter(complete_list_tags)
    print(c.most_common(10))
    print("----------------------------\n")

    ####SUMMARY###
    summary_table = {}
    summary_table['common_tags_names'] = c.most_common(10)
    summary_table['unique_score_types'] = series_scores.unique()
    summary_table['n_score_types'] = len(series_scores.unique())
    #Company ID
    summary_table['company_id'] = company
    #Number of tickets
    summary_table['n_tickets'] = len(table_g.comp_gradable_id.unique())
    #Number of unique scores
    summary_table['n_scores'] = len(table_g.scoreType.unique())
    summary_table['type_scores'] = table_g.scoreType.unique()
    #Number scores below 90
    below_90 = table_g.qa_score <90
    summary_table['n_scores_below_90'] = below_90.sum()
    #Min, Max, median scores
    summary_table['min_score'] = table_g.qa_score.min()
    summary_table['max_score'] = table_g.qa_score.max()
    summary_table['median_score'] = table_g.qa_score.median()
    #PLOTTING
    fig, ax1 = plt.subplots(nrows=1,ncols=1, figsize = (6,4))
    table_g.qa_score.plot.hist(alpha=0.5, ax = ax1,bins = 50)
    ax1.set_title(company)
    ##END  SUMMARY##
    return summary_table, fig

def extract_senti_ag_us_all(table_comments,column, author_type):
    table_comments= table_comments[table_comments.author_type == author_type]
    senti_feature = table_comments['body'].apply(extract_senti)
    senti_feature =  senti_feature.apply(pd.Series)
    senti_feature['comp_gradable_id'] = table_comments['comp_gradable_id'].copy()
    #senti_feature_mean = senti_feature.groupby('comp_gradable_id').mean()
    #senti_feature_min = senti_feature.groupby('comp_gradable_id').min()

    senti_feature['time'] = table_comments['time'].copy()
    senti_feature = senti_feature.sort_values(by='time')
    senti_feature.drop('time', axis=1, inplace=True)
    senti_feature_first = senti_feature.groupby('comp_gradable_id').first()
    senti_feature_last = senti_feature.groupby('comp_gradable_id').last()


    return (senti_feature_first,senti_feature_last)


def compare_precision_recall_plot(classf1, classf2):


    (clf, X_test, y_test)  = classf1
    probas = clf.predict_proba(X_test)
    y_pred = clf.predict(X_test)
    ap1 = average_precision_score(y_test, probas[:, 1])

    precision, recall, thresholds = precision_recall_curve( y_test,  probas[:, 1], pos_label=1)
    fig, ax1 = plt.subplots(nrows=1,ncols=1, figsize = (6,4))
    ax1.set_xlim([-0.05, 1.05])
    ax1.set_ylim([-0.05, 1.05])
    ax1.set_xlabel('recall')
    ax1.set_ylabel('precision')
    ax1.set_title('Precision-Recall Curve')
    #ax1.axhline(y=bad_tickets.iloc[0], color = 'k',linestyle =':',lw=2)

    (clf, X_test, y_test)  = classf2
    probas = clf.predict_proba(X_test)
    y_pred = clf.predict(X_test)
    ap2 = average_precision_score(y_test, probas[:, 1])

    precision, recall, thresholds = precision_recall_curve( y_test,  probas[:, 1], pos_label=2)
    #fig, ax1 = plt.subplots(nrows=1,ncols=1, figsize = (6,4))
    ax1.plot(recall,precision,lw=1,label='area = %0.2f'% ap1)
    ax1.plot(recall,precision,lw=1,label='area = %0.2f'% ap2)
    ax1.legend(loc="lower right")
    plt.show()
    path_graphs = path_graphs = '/Users/mehdiansari/Desktop/MaestroQA/Data/Figures/'
    #fig.savefig(path_graphs+model_name+'precision_recall.pdf')
    fig.savefig(path_graphs+'compare1_2'+'precision_recall.png')
