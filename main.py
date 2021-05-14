import uvicorn
from fastapi import FastAPI
import pickle
import os
import pandas as pd
import numpy as np
from pydantic import BaseModel
#from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_validate 
from sklearn.naive_bayes import GaussianNB
import re
import  nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


app = FastAPI()


# Load model
pickle_file_open = open("cls.pkl","rb") # open pickle file in read mode
model = pickle.load(pickle_file_open) # to load the pickle file

# loading dataset with userId

@app.get('/')
def home():
    return "Welcome All! open FastAPI and input these values in the Post method: EXP - User Text and Rating(4,5) and you will get CPI- Crop Productivity Index"



@app.post('/predict')
def predict(Text:'str'):
    def bag_of_words(your_review):
        dataset = pd.read_csv("chrome_reviews.csv")
        dataset_1 = dataset[dataset.Star == 1]
        dataset_2 = dataset[dataset.Star == 2]
        df_Neg = pd.concat([dataset_1, dataset_2] ,axis=0)
        df_Neg["Review"] = 0
        #df_Neg.head()

        dataset_4 = dataset[dataset.Star == 4]
        dataset_5 = dataset[dataset.Star == 5]
        df_post = pd.concat([dataset_4, dataset_5] ,axis=0)
        #df_post = pd.concat([df_post, dataset_5] ,axis=0)
        df_post["Review"] = 1

        dataset_new = pd.concat([df_Neg ,df_post] ,axis=0)
        dataset_new = dataset_new.reset_index()


        
        
        text_list = list(dataset_new["Text"])
        new_list = text_list + your_review
        print(len(text_list))
        print(len(new_list))
        #print(new_list[-1])
        df1 = {'Text': new_list}
        dataset_new_added = pd.DataFrame(df1)
        print(dataset_new_added['Text'][6753])
        
        corpus = []
        for i in range(0, 6753):
            # Keeping only alphabets
            review = re.sub('[^a-zA-Z]', ' ', dataset_new_added['Text'][i] )
            # All aplhabets must be lower case only
            review = review.lower()
            # Split review in different words
            review = review.split()
            # Remove non-significant words     set is used when review is large like an article
            # Stemming - only keep root word
            ps = PorterStemmer()
            review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
            #  Convert the list review back to string 
            review = ' '.join(review)
            # Append the cleaned review to list corpus
            corpus.append(review)
        print(corpus[-1])
        
        from sklearn.feature_extraction.text import CountVectorizer
        cv = CountVectorizer(max_features=2000)
        X = cv.fit_transform(corpus).toarray()
        print(len(corpus))
        print(len(X))
        result = model.predict(X)
        return (f"The predicted CPI(Crop Productivity Index) is {result[-1]}")
        # print(result)


if __name__=="__main__":
    #port = int(os.environ.get("PORT",8000))
    uvicorn.run(app, host='127.0.0.1', port=5000)