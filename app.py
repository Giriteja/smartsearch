#Usage: python app.py
import os
from flask import Flask, render_template, request, redirect, url_for
import numpy as np
import imutils
import time
import uuid
import base64
from bs4 import BeautifulSoup
import re
import pickle
import time
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, CrossEncoder, util
import os
import pandas as pd
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")
import flask
app = Flask(__name__)

times=[]
names=[]

#We use the Bi-Encoder to encode all passages, so that we can use it with sematic search
bi_encoder = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')
#bi_encoder.max_seq_length = 256     #Truncate long passages to 256 tokens
top_k = 32                          #Number of passages we want to retrieve with the bi-encoder

#The bi-encoder will retrieve 100 documents. We use a cross-encoder, to re-rank the results list to improve the quality
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

# As dataset, we use Simple English Wikipedia. Compared to the full English wikipedia, it has only
# about 170k articles. We split these articles into paragraphs and encode them with the bi-encoder

@app.route('/')
def index():
    return flask.render_template('index.html')



@app.route('/', methods=['GET', 'POST'])
def predict():
    to_predict_list = request.form.to_dict()
    review_text = to_predict_list['review_text']
    
    data_tf=[]
    DF_text_time = pd.read_pickle(r"C:\Newdownloads\Covid19--master\Covid19--master\Transcript\final_sent")
    embd=list(DF_text_time['Embeddings'])
    total_text_set_list=list(DF_text_time['Sentance_3_cleaned'])
    filename=list(DF_text_time['filename'])
    time=list(DF_text_time['time'])
    
    for i in embd:
        data_tf.append(torch.from_numpy(i))
    query = review_text
    
    ##### Sematic Search #####
    # Encode the query using the bi-encoder and find potentially relevant passages
    question_embedding = bi_encoder.encode(query, convert_to_tensor=True)
    question_embedding = question_embedding
    hits = util.semantic_search(question_embedding, data_tf, top_k=10)
    hits = hits[0]  # Get the hits for the first query
    print(hits)
    ##### Re-Ranking #####
    # Now, score all retrieved passages with the cross_encoder
    cross_inp = [[query, total_text_set_list[hit['corpus_id']]] for hit in hits]
    cross_scores = cross_encoder.predict(cross_inp)
    
    # Sort results by the cross-encoder scores
    for idx in range(len(cross_scores)):
        hits[idx]['cross-score'] = cross_scores[idx]
    
    # Output of top-5 hits from bi-encoder
    print("\n-------------------------\n")
    print("Top-3 Bi-Encoder Retrieval hits")
    hits = sorted(hits, key=lambda x: x['score'], reverse=True)
    for hit in hits[0:50]:
        print("\t{:.3f}\t{}".format(hit['score'], total_text_set_list[hit['corpus_id']]))
    
    # Output of top-5 hits from re-ranker
    print("\n-------------------------\n")
    print("Top-3 Cross-Encoder Re-ranker hits")
    hits = sorted(hits, key=lambda x: x['cross-score'], reverse=True)
    
    for i in range(len(hits)):
        if(i!=0 and abs(int(time[hits[i]['corpus_id']])-int(time[hits[i-1]['corpus_id']]))>=10):
            names.append(filename[hits[i]['corpus_id']][:-4])
            times.append(int(time[hits[i]['corpus_id']]))
        if(i==0):
            names.append(filename[hits[i]['corpus_id']][:-4])
            times.append(int(time[hits[i]['corpus_id']]))
    for hit in hits[0:50]:
        print(filename[hit['corpus_id']][:-4], total_text_set_list[hit['corpus_id']],time[hit['corpus_id']])
        
    return render_template('template.html', label=names,time=times)


if __name__ == '__main__':
    app.debug=False
    app.run()