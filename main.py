import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import string
import openpyxl
import nltk
nltk.download('all')
from nltk import tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score,precision_score,recall_score,confusion_matrix,roc_curve,classification_report
import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)

from YontemML import YontemML
from modelTuning import OptimizeModel
from handlingImbalancedData import ImbalanceDuzenle
from wordcloudapp import wordclouddraw

data=pd.read_excel("data/processedData.xlsx")
data.head()

#Review: Review text

#Rating: Given score (1-5)

#Total_thumbsup: How many people found the review helpful



data.isnull().sum()

data.Rating.value_counts()

plt.figure(figsize=(15,6))
sns.countplot(x=data.Rating, data=data)
plt.show()