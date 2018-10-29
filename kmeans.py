
import pandas as pd
import numpy as np
import codecs
from nltk.corpus import stopwords
import imp
from nltk.tokenize import word_tokenize


name=[]
for i in range(30):
    s = "file"+ str(i) +".txt" 
    name.append(s)


stop_words = set(stopwords.words('english'))
main =[]
i=0
for item in name:
    w=""
    s=""
    file1 = codecs.open(item, encoding='utf-8')
    word_tokens = word_tokenize(file1.read())
    for w in word_tokens:
        if w not in stop_words:
            s = s +" "+w
    main.append(s)

    
#Count Vectoriser then tidf transformer

from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(main)

#vectorizer.get_feature_names()

#print(X.toarray())     

from sklearn.feature_extraction.text import TfidfTransformer

transformer = TfidfTransformer(smooth_idf=False)
tfidf = transformer.fit_transform(X)
print(tfidf.shape)                        

from sklearn.cluster import KMeans
#Change it according to your data.
num_clusters = 5 
km = KMeans(n_clusters=num_clusters)
km.fit(tfidf)
clusters = km.labels_.tolist()
#Creating dict having doc with the corresponding cluster number.
idea={'Idea':main, 'Cluster':clusters} 
# Converting it into a dataframe.
frame=pd.DataFrame(idea,index=[clusters], columns=['Idea','Cluster']) 

print("\n")
print(frame) 
#Print the doc with the labeled cluster number.
print("\n")
#Print the counts of doc belonging to each cluster.
print(frame['Cluster'].value_counts()) 

