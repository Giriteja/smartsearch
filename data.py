import json
from sentence_transformers import SentenceTransformer, CrossEncoder, util
import gzip
import os
import torch
from youtube_transcript_api import YouTubeTranscriptApi
from tqdm import tqdm
import re

if not torch.cuda.is_available():
    print("Warning: No GPU found. Please add GPU to your notebook")


#We use the Bi-Encoder to encode all passages, so that we can use it with sematic search
bi_encoder = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')
#bi_encoder.max_seq_length = 256     #Truncate long passages to 256 tokens
top_k = 32                          #Number of passages we want to retrieve with the bi-encoder

#The bi-encoder will retrieve 100 documents. We use a cross-encoder, to re-rank the results list to improve the quality
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

# As dataset, we use Simple English Wikipedia. Compared to the full English wikipedia, it has only
# about 170k articles. We split these articles into paragraphs and encode them with the bi-encoder

from bs4 import BeautifulSoup


def decontracted(phrase):
    # specific
    phrase = re.sub(r"won't", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)

    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase
    
def clean_code(sentance):
    sentance=sentance.replace("\n","")
    sentance = re.sub(r"http\S+", "", sentance)
    sentance = BeautifulSoup(sentance, 'lxml').get_text() #python-beautifulsoup-how-to-remove-all-tags-from-an-element
    sentance = decontracted(sentance)
    sentance = re.sub("\S*\d\S*", "", sentance).strip() #remove words with numbers python
    sentance = re.sub('[^A-Za-z]+', ' ', sentance) #remove special character
    # https://gist.github.com/sebleier/554280
    sentance = ' '.join(e.lower() for e in sentance.split())
    return sentance

def generate_N_grams(sentences,ngram=1):
  words=[clean_code(sentence) for sentence in sentences]  
  temp=zip(*[words[i:] for i in range(0,ngram)])
  ans=[' '.join(ngram) for ngram in temp]
  return ans

import os
from tqdm import tqdm
from nltk.corpus import words
import pandas as pd
from youtube_transcript_api import YouTubeTranscriptApi

DF_text_time=pd.DataFrame(columns=['Sentance_3_cleaned','Sentance_3','Embeddings','time','filename'])
text=[]
start=[]
n_3_grams=[]
total_emb=[]
sentences=[]
filename=[]
corpus_sentences=[]
count=0
path = r"C:\Newdownloads\Covid19--master\Covid19--master\Transcript"
for dir,subdir,files in os.walk(path):
    for filename in tqdm(files):
        df = pd.DataFrame(columns = ['Sentance_3_cleaned','Sentance_3','Embeddings',"time",'filename'])
        json=YouTubeTranscriptApi.get_transcript(filename)
        for i in json:
          text.append(i['text'])
          start.append(i['start'])
        n_3_grams=generate_N_grams(text,3)
        for sen in n_3_grams:
          total_emb.append(bi_encoder.encode(sen,convert_to_numpy=True))

        df['Sentance_3_cleaned']=n_3_grams
        df['Embeddings']=total_emb
        df['Sentance_3']=text[:-2]
        df['time']=start[:-2]
        df['filename']=filename
        DF_text_time=DF_text_time.append(df)
        n_3_grams=[]
        total_emb=[]
        text=[]
        start=[]
        del df
DF_text_time.to_pickle(r'C:\Newdownloads\Covid19--master\Covid19--master\Transcript\final_sent')

# We encode all passages into our vector space. This takes about 5 minutes (depends on your GPU speed)
#corpus_embeddings = bi_encoder.encode(total_text, convert_to_tensor=True, show_progress_bar=True)