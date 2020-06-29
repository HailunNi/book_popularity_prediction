#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 13:40:10 2020

@author: maomao
"""

#%%
import streamlit as st
import numpy as np
import pandas as pd
import pickle

import textwrap
import transformers
import pickle
import torch
import time
import datetime
import numpy as np
import faiss
import matplotlib.pyplot as plt

from transformers import BertForSequenceClassification, BertTokenizer
from keras.preprocessing.sequence import pad_sequences
    
#%% Load the XGBoost classification model
# import xgboost

# clf = xgboost.Booster({'nthread': 4})  # init model
# clf.load_model('./model_save/alldata_xgboost_clf.model')  # load data

#%% Load the author info dataframe
authors = pickle.load(open('../data/model_save/author_dict.sav','rb'))

#%%
st.write("""
         # Fantasy Novel Popularity Predictor
         *Will your book beat the market median?*
         """)

# author_name = st.text_input("Author name", "Henry Eason")
# author_name = st.text_input("Author name", "J.R.R. Tolkien")

st.write("""
         ## User Inputs
         """)

author_name = st.text_input("Author name", "Nalini Singh")
is_series = st.checkbox("Part of a series?", value=True)      
pages = st.number_input("Number of pages", value=368)

# author_name = st.text_input("Author name", "J.K. Rowling")
# is_series = st.checkbox("Part of a series?", value=True)      
# pages = st.number_input("Number of pages", value=500)
year = st.slider("Publication year",2000,2021,2020)

#%%
# query_text = st.text_area('Synopsis',"In the land of magic, a young boy is destined for greatness.")
# query_text = st.text_area('Synopsis',"In a world where virtue means life, if you run out of virtue points you meet certain death.")
# query_text = st.text_area('Synopsis',"in a world where survival depends on points gathered through virtuous acts, nobel prize winners are filthy rich and criminals are dirt poor. a slum boy woke up one day realizing he alone had the ability to transfer points between people. now the boy's adventure began.")
query_text = st.text_area('Synopsis',"New York City, 1899. Tillie Pembroke’s sister lies dead, her body drained of blood and with two puncture wounds on her neck. Bram Stoker’s new novel, Dracula, has just been published, and Tillie’s imagination leaps to the impossible: the murderer is a vampire. But it can’t be—can it? A ravenous reader and researcher, Tillie has something of an addiction to truth, and she won’t rest until she unravels the mystery of her sister’s death. Unfortunately, Tillie’s addicted to more than just truth; to ease the pain from a recent injury, she’s taking more and more laudanum…and some in her immediate circle are happy to keep her well supplied. Tillie can’t bring herself to believe vampires exist. But with the hysteria surrounding her sister’s death, the continued vampiric slayings, and the opium swirling through her body, it’s becoming increasingly difficult for a girl who relies on facts and figures to know what’s real—or whether she can trust those closest to her.")

run_model = st.button("Predict novel popularity")



#%%
#%%
#%%
df = pickle.load(open('../data/model_save/df.sav','rb'))

#%%
df_metadata = df[['series','num_pages','author_rating','publication_year']] # Best model
y = df['labels']

# from sklearn.model_selection import train_test_split, cross_val_score
# X_train, X_test, y_train, y_test = train_test_split(df_metadata,y,test_size=0.2, random_state=2020)

# #%
bert_train_predictions = pickle.load(open('../data/model_save/train_predictions.sav','rb'))

# #%
df_metadata['BERT_result'] = bert_train_predictions[:,1]

#%
from xgboost import XGBClassifier
clf = XGBClassifier()
clf.fit(df_metadata, y)

# from sklearn.ensemble import AdaBoostClassifier
# clf = AdaBoostClassifier() 
# clf.fit(X_train, y_train)


#%%
@st.cache 
def load_bert_model():
    # The name of the folder containing the model files.
    output_dir = '../data/model_save/model_save/'
    
    # Load our fine-tuned model, and configure it to return the "hidden states", 
    # from which we will be taking our text embeddings.
    model = BertForSequenceClassification.from_pretrained(
        output_dir,
        output_hidden_states = True, # Whether the model returns all hidden-states.
    ) 
    
    # Load the tokenizer.
    tokenizer = BertTokenizer.from_pretrained(output_dir)
    return model, tokenizer

model,tokenizer = load_bert_model()

#%%
@st.cache 
def text_to_embedding(tokenizer, model, in_text):
    '''
    Uses the provided BERT `model` and `tokenizer` to generate a vector 
    representation of the input string, `in_text`.

    Returns the vector stored as a numpy ndarray.
    '''

    # ===========================
    #    STEP 1: Tokenization
    # ===========================

    MAX_LEN = 300 # 128

    # `encode` will:
    #   (1) Tokenize the sentence.
    #   (2) Truncate the sentence to MAX_LEN if necessary.
    #   (2) Prepend the `[CLS]` token to the start.
    #   (3) Append the `[SEP]` token to the end. (After truncating!)
    #   (4) Map tokens to their IDs.
    input_ids = tokenizer.encode(
                        in_text,                    # Sentence to encode.
                        add_special_tokens = True,  # Add '[CLS]' and '[SEP]'
                        max_length = MAX_LEN,       # Truncate all sentences.                        
                   )    

    # Pad our input tokens. Truncation was handled above by the `encode`
    # function, which also makes sure that the `[SEP]` token is placed at the
    # end *after* truncating.
    # Note: `pad_sequences` expects a list of lists, but we only have one
    # piece of text, so we surround `input_ids` with an extra set of brackets.
    results = pad_sequences([input_ids], maxlen=MAX_LEN, dtype="long", 
                              truncating="post", padding="post")
    
    # Remove the outer list.
    input_ids = results[0]

    # Create attention masks    
    attn_mask = [int(i>0) for i in input_ids]
    
    # Cast to tensors.
    input_ids = torch.tensor(input_ids)
    attn_mask = torch.tensor(attn_mask)

    # Add an extra dimension for the "batch" (even though there is only one 
    # input in this batch.)
    input_ids = input_ids.unsqueeze(0)
    attn_mask = attn_mask.unsqueeze(0)

    # ===========================
    #    STEP 2: BERT Model
    # ===========================

    # Put the model in evaluation mode--the dropout layers behave differently
    # during evaluation.
    model.eval()

    # Copy the inputs to the GPU
    # Note -- I got stuck here for a while because I didn't assign the result 
    # back to the variable! Geez!
    # input_ids = input_ids.to(device)
    # attn_mask = attn_mask.to(device)
    
    # Telling the model not to build the backwards graph will make this 
    # a little quicker.
    with torch.no_grad():        

        # Forward pass, return hidden states and predictions.
        # This will return the logits rather than the loss because we have
        # not provided labels.
        logits, encoded_layers = model(
                                    input_ids = input_ids, 
                                    token_type_ids = None, 
                                    attention_mask = attn_mask)
        
    # Retrieve our sentence embedding--take the `[CLS]` embedding from the final
    # layer.
    layer_i = 12 # The last BERT layer before the classifier.
    batch_i = 0 # Only one input in the batch.
    token_i = 0 # The first token, corresponding to [CLS]
        
    # Grab the embedding.
    vec = encoded_layers[layer_i][batch_i][token_i]

    # Move to the CPU and convert to numpy ndarray.
    # vec = vec.detach().cpu().numpy()
    vec = vec.numpy()

    return logits,vec

#%%
# def format_time(elapsed):
#     '''
#     Takes a time in seconds and returns a string hh:mm:ss
#     '''
#     # Round to the nearest second.
#     elapsed_rounded = int(round((elapsed)))
    
#     # Format as hh:mm:ss
#     return str(datetime.timedelta(seconds=elapsed_rounded))

#%%
# vecs = np.load('./model_save/embeddings.npy')
# # vecs.shape

#%%
# # =====================================
# #            FAISS Setup
# # =====================================

# # @st.cache 
# def faiss_setup():
#     # Build a flat (CPU) index
#     cpu_index = faiss.IndexFlatL2(vecs.shape[1]) # using Euclidean distance here. You can also use cosine!!!
#     # cpu_index = faiss.IndexFlatIP(vecs.shape[1])
#     cpu_index.add(vecs)
#     return cpu_index

# cpu_index = faiss_setup()

#%%
@st.cache 
def bert_prediction(tokenizer, model, query_text):
    # Vectorize a new piece of text.
    pred,query_vec = text_to_embedding(tokenizer, model, query_text)
    return pred,query_vec
    
#%%
# # Use `textwrap` to print the sentence nicely.
# wrapper = textwrap.TextWrapper(initial_indent="    ", subsequent_indent="    ", 
#                                width = 80)

# # @st.cache 
# def get_similar_books(query_vec):
#     # Let's find the 5 most similar books.
#     D, I = cpu_index.search(query_vec.reshape(1, 768), k=5) 
    
#     st.write('')
#     st.write('==== Top 5 Similar Title Results that are Popular ====')
    
#     # For each result...
#     # tally = 1
#     for i in range(5):
#         # if tally > 5:
#             # break
        
#         # Look up the comment row number for this result.
#         result_i = I[0, i]
        
#         # if df.iloc[result_i]['labels']:
#         # Look up the text for this comment.
#         # text = df_synopsis.iloc[result_i].description
#         text = df.iloc[result_i]['description']
    
#         # st.write('Comment #{:,}:'.format(result_i))
#         # st.write('L2 Distance: %.2f' % D[0, i])
#         st.image(df.iloc[result_i]['image_url'])
#         st.write(df.iloc[result_i]['title'])
#         st.write(df.iloc[result_i]['publisher'])
#         st.write(int(df.iloc[result_i]['publication_year']))
#         # st.write('Rating: ', df.iloc[result_i]['average_rating'])
#         st.write('Popular: ', df.iloc[result_i]['labels'])
#         st.write(wrapper.fill('"' + text + '"'))
#         st.write('')
        
#             # tally += 1

# def get_similar_books(query_vec):
#     # Let's find the 5 most similar books.
#     D, I = cpu_index.search(query_vec.reshape(1, 768), k=100) 
    
#     st.write('')
#     st.write('==== Top 5 Similar Title Results that are Popular ====')
    
#     # For each result...
#     tally = 1
#     for i in range(100):
#         if tally > 5:
#             break
        
#         # Look up the comment row number for this result.
#         result_i = I[0, i]
        
#         if df.iloc[result_i]['labels']:
#             # Look up the text for this comment.
#             # text = df_synopsis.iloc[result_i].description
#             text = df.iloc[result_i]['description']
        
#             # st.write('Comment #{:,}:'.format(result_i))
#             # st.write('L2 Distance: %.2f' % D[0, i])
#             st.image(df.iloc[result_i]['image_url'])
#             st.write(df.iloc[result_i]['title'])
#             st.write(df.iloc[result_i]['publisher'])
#             st.write(int(df.iloc[result_i]['publication_year']))
#             # st.write('By',df.iloc[result_i]['author_name'])
#             # st.write('Rating: ', df.iloc[result_i]['average_rating'])
#             # st.write('Popular: ', df.iloc[result_i]['labels'])
#             st.write(wrapper.fill('"' + text + '"'))
#             st.write('')
        
#             tally += 1

#%%
if run_model:
    st.write("""
             ## Model Outputs
             """)
    try:
        author_rating = authors[author_name]
        st.write("Author rating found: ", author_rating)
    except:
        author_rating = 4 # needs to be the average!
        st.write("Author rating not found, assumed to be the average: ", author_rating)
        
    bert_predicted,query_vec = bert_prediction(tokenizer, model, query_text)
    # st.write(bert_predicted)
    inputs = pd.DataFrame([[int(is_series),pages,float(author_rating),year,float(bert_predicted[:,1])]],
                          columns=['series','num_pages','author_rating','publication_year','BERT_result'])
    # inputs = [[int(is_series),pages,author_rating,year,bert_predicted[:,1]]]
    # st.write(inputs)
    y_predicted = clf.predict(inputs)
    
    # st.write(y_predicted)
    
    if y_predicted:
        st.write("Hurray! It looks like this book will be popular!")
        # st.write("Below are similar books that are also popular.")
    else:
        st.write("Uh-oh! It looks like this book may not be popular.")
        # st.write("Below are similar books that are popular for your reference.")
    # get_similar_books(query_vec)
    
#%%
# year = 2020
# pages = 300
# is_series = True
# author_rating = 4
# bert_predicted = [-1,1]
# inputs = pd.DataFrame([[int(is_series),pages,author_rating,year,bert_predicted[1]]],
#                       columns=['series','num_pages','author_rating','publication_year','BERT_result'])