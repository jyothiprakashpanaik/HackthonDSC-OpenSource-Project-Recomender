#!/usr/bin/env python
# coding: utf-8

# In[19]:


import pickle
import pandas as pd
#import nltk
from nltk import word_tokenize 
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class LemmaTokenizer:
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self,articles):
        return [self.wnl.lemmatize(t) for t in word_tokenize(articles)]


# Global Variables

repo_lookup = pickle.load(open("models/repo_lookup.pkl","rb"))
stopwords = pickle.load(open("models/stopwords.pkl","rb"))
word_vectorizer = CountVectorizer(ngram_range=(1,1), analyzer='word',stop_words=stopwords, tokenizer=LemmaTokenizer())
sparse_vector_matrix = word_vectorizer.fit_transform(repo_lookup['repo_description_plus_topic'])
    


# In[16]:



def text_recommender(input_df, word_vectorizer=word_vectorizer,  sparse_vector_matrix = sparse_vector_matrix, repo_lookup=repo_lookup):
    
    input_df["bag_of_words"] = input_df.apply(lambda x: " ".join(x), axis=1)
    
    # vectorize the inputted string
    inputted_vector = word_vectorizer.transform(input_df['bag_of_words'])
    
    # calculate the cosine similarity
    one_dimension_cosine_sim = cosine_similarity(inputted_vector, sparse_vector_matrix)
    
    # creating a Series with the similarity scores in descending order
    score_series = pd.Series(one_dimension_cosine_sim[0]).sort_values(ascending = False)
    
    # only show matches that have some similarity
    score_series = score_series[score_series>0]
    
    # getting the indexes of the 10 most similar repos
    top_10_indexes = list(score_series.iloc[1:10].index)
    
    # return the list of recommended repo
    recommended_repos = repo_lookup.loc[top_10_indexes]
    
    return recommended_repos


# In[17]:



class RepoRecomender(object):
  
    def __init__(self,word_vectorizer,sparse_vector_matrix,repo_lookup,text_recommender):
        self.word_vectorizer = word_vectorizer 
        self.sparse_vector_matrix = sparse_vector_matrix
        self.repo_lookup = repo_lookup
        self.text_recommender = text_recommender
        
    def predict(self,context,model_input):
        output_df = self.text_recommender(model_input)
        return [output_df.to_dict('records')]
    


# In[21]:

def nltk_model_predict(model_input):
    model = RepoRecomender(word_vectorizer = word_vectorizer,
                                           sparse_vector_matrix = sparse_vector_matrix,
                                           repo_lookup = repo_lookup,
                                           text_recommender = text_recommender)
    model_output = model.predict(None,model_input)
    return model_output

if __name__=="__main__":
#     model = RepoRecomender(word_vectorizer = word_vectorizer,
#                                            sparse_vector_matrix = sparse_vector_matrix,
#                                            repo_lookup = repo_lookup,
#                                            text_recommender = text_recommender)

#     model_input = pd.DataFrame([["Java","Kotlin", "Android"]])
#     model_output = model.predict(None,model_input)

#     print(model_output)
    model_input = pd.DataFrame([["Java","Kotlin", "Android"]])
    model_output = nltk_model_predict(model_input)
    print(model_output)

# In[ ]:




