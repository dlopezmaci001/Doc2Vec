# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 16:16:55 2019

@author: daniel.lopez
"""

#Import all the dependencies
import gensim
from nltk import RegexpTokenizer
from nltk.corpus import stopwords
from os import listdir
from os.path import isfile, join
import pandas as pd

#now create a list that contains the name of all the text file in your data #folder
docslabels = []
docslabels = [f for f in listdir(r'C:\Users\daniel.lopez\Desktop\IE\Facturas') if f.endswith('.txt')]

#create a list data that stores the content of all text files in order of their names in docLabels
data = []
for doc in docslabels:
    data.append(open(r'C:\Users\daniel.lopez\Desktop\IE\Facturas\\' + doc, encoding = 'latin-1').read())
    
# Remove characters from doc lables
docLabels = []

for name in docslabels:
    name = name.replace('.txt','') 
    name = name.lower()
    docLabels.append(name)


tokenizer = RegexpTokenizer(r'\w+')
stopword_set = set(stopwords.words('spanish'))

#This function does all cleaning of data using two objects above
def nlp_clean(data):
   new_data = []
   for d in data:
      new_str = d.lower()
      dlist = tokenizer.tokenize(new_str)
      dlist = list(set(dlist).difference(stopword_set))
      new_data.append(dlist)
   return new_data

class LabeledLineSentence(object):
    def __init__(self, doc_list, labels_list):
        self.labels_list = labels_list
        self.doc_list = doc_list
    def __iter__(self):
        for idx, doc in enumerate(self.doc_list):
              yield gensim.models.doc2vec.LabeledSentence(doc,    
[self.labels_list[idx]])
              
data = nlp_clean(data)              

#iterator returned over all documents
it = LabeledLineSentence(data, docLabels)

model = gensim.models.Doc2Vec(size=300, min_count=0, alpha=0.025, min_alpha=0.025)
model.build_vocab(it)
#training of model
for epoch in range(100):
 print ('iteration '+str(epoch+1))
 model.train(it,total_examples=model.corpus_count,epochs=epoch)
 model.alpha -= 0.002
 model.min_alpha = model.alpha
#saving the created model
model.save('doc2vec.model')
print ('model saved')

#loading the model
d2v_model = gensim.models.doc2vec.Doc2Vec.load('doc2vec.model')

#start testing
#printing the vector of document at index 1 in docLabels
docvec = pd.np.mean(d2v_model.docvecs[1])
print(docvec)

avg_list= list()

for i in range(0,len(d2v_model.docvecs)):
    avg_np = (d2v_model.docvecs[i])
    avg_list.append(avg_np)
    
df = pd.DataFrame()
    
df['name'] = pd.Series(docLabels) 

df['avg_np'] = pd.Series(avg_list) 

# Creating the annoy 
from annoy import AnnoyIndex

# Number of dimensions of the vector annoy is going to store. 
# Make sure it's the same as the word2vec we're using!
f = 300

# Specify the metric to be used for computing distances. 
u = AnnoyIndex(f, metric='angular') 

df['index'] = [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15]

# We can sequentially add items.
for index,row in df.iterrows():
    u.add_item(row['index'],row['avg_np'])

# Number of trees for queries. When making a query the more trees the easier it is to go down the right path. 
u.build(10000) # 10 trees

data_nocorpus = []
data_nocorpus.append(open(r'C:\Users\daniel.lopez\Desktop\IE\Facturas\(36) litolux 19091 SPOON INSIGHT S.L.txt', encoding = 'latin-1').read())
docvec_nosee = d2v_model.infer_vector(data_nocorpus)

#similar_products = u.get_nns_by_item(10, 10)
similar_facts = u.get_nns_by_vector(docvec_nosee, 5, search_k=-1, include_distances=True)

print(similar_facts)
