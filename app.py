import pandas as pd
import streamlit as st
from PIL import Image
import pandas as pd
import numpy as np
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk import PorterStemmer, WordNetLemmatizer
from Prediction import *
import pickle

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('omw-1.4')

# Page title
st.set_page_config(page_title="Chintamani Chourase", page_icon=Image.open('statics/Me.jpeg'),initial_sidebar_state="expanded")

# Add profile image
profile_image = Image.open('statics/Me.jpeg')
st.sidebar.image(profile_image, use_column_width=True)

# Add contact information
st.sidebar.title("Chintamani Chourase")
st.sidebar.write("Data Scientist & AI Engineer")
st.sidebar.write("You can reach me at:")
st.sidebar.subheader("chintamanichourase@gmail.com")
st.sidebar.subheader("[LinkedIn](https://www.linkedin.com/in/chintamani-chourase-43964122b/)")
st.sidebar.subheader("[Instagram](http://instagram.com/disposable_account02)")
st.sidebar.subheader("[GitHub](https://github.com/Chintamanichourase)")
st.sidebar.subheader("[Kaggle](https://www.kaggle.com/chintamanichourase)")

#Skills
st.sidebar.header("Skills")
st.sidebar.write("Here are some of my top skills:")
st.sidebar.write("- Python, SQL")
st.sidebar.write("- Databases : PostgreSQL, MongoDB, Pinecone, Qdrant")
st.sidebar.write("- Libraries & Frameworks : FastAPI, Openai, Llama-index, Langchain, Pinecone, LLM-Sherpa, Openpipe, PyMongo, Transformers, Huggingface, Boto3, Pandas, Numpy, Pandasai, Matplotlib, Seaborn, Scikit-Learn, Streamlit, BeautifulSoup")
st.sidebar.write("- Models : GPT-3.5, GPT-4, Text-da-vinci, Dall.E,Instructor Large, Llama-2, LLama-3, Stable Diffusion, Bart-Base, Claude Instant V1, Claude 3, Phi-3")
st.sidebar.write("- Data Science : Data Collection, Data Wrangling, Data Visualization, Exploratory Data Analysis, Feature Engineering, Feature Selection, Machine Learning(Regression, Classification, Clustering), Model Evaluation, Model Deployment")
st.sidebar.write("- AWS Services : Bedrock, Textract, Cognito")
st.sidebar.write("- Version Control & Deployment : Git & Github")



st.write('''
# Cyberbullying Tweet Recognition App

This app predicts the nature of the tweet into 6 Categories.
* Age
* Ethnicity
* Gender
* Religion
* Other Cyberbullying
* Not Cyberbullying

***
''')

image = Image.open('statics/twitter.png')
st.image(image, use_column_width= True)

# Text Box
st.header('Enter Tweet ')
tweet_input = st.text_area("Tweet Input", height= 150)
print(tweet_input)
st.write('''
***
''')

# print input on webpage
if tweet_input:
    st.header('''
    ***Predicting......
    ''')
else:
    st.write('''
    ***No Tweet Text Entered!***
    ''')
st.write('''
***
''')

# Output on the page
st.header("Prediction")
if tweet_input:
    prediction = prediction(tweet_input)
    if prediction == "age":
        st.image("statics/Age.png",use_column_width= True)
    elif prediction == "ethnicity":
        st.image("statics/Ethnicity.png",use_column_width= True)
    elif prediction == "gender":
        st.image("statics/Gender.png",use_column_width= True)
    elif prediction == "other_cyberbullying":
        st.image("statics/Other.png",use_column_width= True)
    elif prediction == "religion":
        st.image("statics/Religion.png",use_column_width= True)
    elif prediction == "not_cyberbullying":
        st.image("statics/not_cyber.png",use_column_width= True)
else:
    st.write('''
    ***No Text Entered!***
    ''')

st.write('''***''')
