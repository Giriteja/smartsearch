from streamlit_player import st_player
import streamlit as st
import numpy as np
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

st.set_page_config(layout="wide")

warnings.filterwarnings("ignore")

times=[]
names=[]

col1, col2 = st.columns([2,1])

@st.cache(allow_output_mutation=True)
def model():   
    #We use the Bi-Encoder to encode all passages, so that we can use it with sematic search
    bi_encoder = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')
    #bi_encoder.max_seq_length = 256     #Truncate long passages to 256 tokens
    top_k = 32                          #Number of passages we want to retrieve with the bi-encoder
    
    #The bi-encoder will retrieve 100 documents. We use a cross-encoder, to re-rank the results list to improve the quality
    cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    
    # As dataset, we use Simple English Wikipedia. Compared to the full English wikipedia, it has only
    # about 170k articles. We split these articles into paragraphs and encode them with the bi-encoder
    return bi_encoder,cross_encoder
@st.cache    
def data():
    DF_text_time = pd.read_pickle(r"final_sent")
    embd=list(DF_text_time['Embeddings'])
    total_text_set_list=list(DF_text_time['Sentance_3_cleaned'])
    filename=list(DF_text_time['filename'])
    time=list(DF_text_time['time'])
    return DF_text_time,embd,total_text_set_list,filename,time



with col1:
    st.write("""# Smart Search""")
    question = st.text_input('Please type your Query') 
    
    if question:
        #01:36 (primary key) 0:05 (visualization) 0:2:6(multiple records)
        #start=int(timedelta(hours=0, minutes=0, seconds=int(time%60)).total_seconds())
        #print(start)
        review_text = question
        
        data_tf=[]
        DF_text_time,embd,total_text_set_list,filename,time=data()
        
        for i in embd:
            data_tf.append(torch.from_numpy(i))
        query = review_text
        
        ##### Sematic Search #####
        # Encode the query using the bi-encoder and find potentially relevant passages
        bi_encoder,cross_encoder=model()
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
            
        minutes0=int(times[0]/60)
        seconds0=int(times[0]%60)
        minutes1=int(times[1]/60)
        seconds1=int(times[1]%60)
        minutes2=int(times[2]/60)
        seconds2=int(times[2]%60)
        minutes3=int(times[3]/60)
        seconds3=int(times[3]%60)
        minutes4=int(times[4]/60)
        seconds4=int(times[4]%60)
        
        st_player("https://www.youtube.com/watch?v="+names[0]+"#t="+str(minutes0)+"m"+str(seconds0)+"s")
        st_player("https://www.youtube.com/watch?v="+names[1]+"#t="+str(minutes1)+"m"+str(seconds1)+"s")
        st_player("https://www.youtube.com/watch?v="+names[2]+"#t="+str(minutes2)+"m"+str(seconds2)+"s")
        st_player("https://www.youtube.com/watch?v="+names[3]+"#t="+str(minutes3)+"m"+str(seconds3)+"s")
        st_player("https://www.youtube.com/watch?v="+names[4]+"#t="+str(minutes4)+"m"+str(seconds4)+"s")
            
    
        with col2:
    
            st.sidebar.title("""Are the Recommendations Helpfull?""")
            
            with open("fileyes.txt", "r") as f:
                a = f.readline()  # starts as a string
                a = 0 if a == "" else int(a)  # check if its an empty string, otherwise should be able to cast using int()
                
            with open("fileno.txt", "r") as f:
                b = f.readline()  # starts as a string
                b = 0 if b == "" else int(b)  # check if its an empty string, otherwise should be able to cast using int()
        
            if st.sidebar.button("Yes"):
                a += 1  
                with open("fileyes.txt", "w") as f:
                    f.truncate()
                    f.write(f"{a}")
                with open("fileno.txt", "r") as fn:
                    n = fn.readline() 
                with open("fileyes.txt", "r") as f:
                    y = f.readline()
                st.write("""## Metrics""")
                st.write("""Yes:""",y)
                st.write("""No:""",n)
                st.write("Score:",int(y)/(int(y)+int(n)))
                    
            if st.sidebar.button("No"):
                b += 1  
                with open("fileno.txt", "w") as f:
                    f.truncate()
                    f.write(f"{b}")
                with open("fileyes.txt", "r") as fy:
                    y = fy.readline() 
                with open("fileno.txt", "r") as f:
                    n = f.readline()
                st.write("""## Metrics""")
                st.write("""Yes:""",y)
                st.write("""No:""",n)
                st.write("Score:",int(y)/(int(y)+int(n)))
