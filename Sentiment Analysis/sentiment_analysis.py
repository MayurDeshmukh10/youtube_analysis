import re
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
import string
import nltk
import warnings 
warnings.filterwarnings("ignore", category=DeprecationWarning)


train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

train['Comment']=train['Comment'].fillna("")
test['Comment']=test['Comment'].fillna("")

print(train.head())

combi = train.append(test, ignore_index=True)

combi['Comment'] = combi['Comment'].str.replace("[^a-zA-Z#]", " ")

combi['Comment'] = combi['Comment'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))

combi.head()


tokenized_comment = combi['Comment'].apply(lambda x: x.split())

tokenized_comment.head()


from nltk.stem.porter import *
stemmer = PorterStemmer()

tokenized_comment = tokenized_comment.apply(lambda x: [stemmer.stem(i) for i in x]) # stemming
tokenized_comment.head()

for i in range(len(tokenized_comment)):
    tokenized_comment[i] = ' '.join(tokenized_comment[i])


combi['Comment'] = tokenized_comment

print(combi['Comment'])

all_words = ' '.join([text for text in combi['Comment']])
from wordcloud import WordCloud
wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(all_words)

plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()


normal_words =' '.join([text for text in combi['Comment'][combi['polarity'] == 1]])

wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(normal_words)
plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()



negative_words = ' '.join([text for text in combi['Comment'][combi['polarity'] == 0]])
wordcloud = WordCloud(width=800, height=500,
random_state=21, max_font_size=110).generate(negative_words)
plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()

from sklearn.feature_extraction.text import CountVectorizer
bow_vectorizer = CountVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english')
# bag-of-words feature matrix
bow = bow_vectorizer.fit_transform(combi['Comment'])

# TF-IDF feature matrix
'''from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english')

tfidf = tfidf_vectorizer.fit_transform(combi['Comment'])'''



from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

train_bow = bow[:17741,:]
test_bow = bow[17742:,:]

#train_tfidf = tfidf[:17741,:]
#test_tfidf = tfidf[17742:,:]


# splitting data into training and validation set
xtrain_bow, xvalid_bow, ytrain, yvalid = train_test_split(train_bow, train['polarity'], random_state=42, test_size=0.3)
#xtrain_bow, xvalid_bow, ytrain, yvalid = train_test_split(train_tfidf, train['polarity'], random_state=42, test_size=0.3)

#xtrain_tfidf = train_tfidf[ytrain.index]
#xvalid_tfidf = train_tfidf[yvalid.index]


#----------------------------------------------------------------------------------------------------

from sklearn import svm

svc = svm.SVC(kernel='linear', C=1, probability=True,decision_function_shape='ovo').fit(xtrain_bow, ytrain)

prediction = svc.predict_proba(xvalid_bow)
prediction_int = prediction[:,1] >= 0.3
prediction_int = prediction_int.astype(np.int)



#--------------------------------------------------------------------------------------------------------------

'''from sklearn.linear_model import LogisticRegression


lreg = LogisticRegression(solver='lbfgs',max_iter=200)

#lreg.fit(xtrain_bow, ytrain) # training the model with bow

lreg.fit(xtrain_tfidf, ytrain)

prediction = lreg.predict_proba(xvalid_bow) # predicting on the validation set
prediction_int = prediction[:,1] >= 0.3 # if prediction is greater than or equal to 0.3 than 1 else 0
prediction_int = prediction_int.astype(np.int)'''
#-----------------------------------------------------------------------------------------------------------------

'''from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=400, random_state=11).fit(xtrain_bow, ytrain)
prediction_int = rf.predict(xvalid_bow)'''

#-----------------------------------------------------------------------------------------------------------------

print("--------------------------------------------Results--------------------------------------------")

print()

print("     F1 Score = ",f1_score(yvalid, prediction_int)) # calculating f1 score
print()
print("     Confusion Matrix of Model")
print()
print(confusion_matrix(yvalid,prediction_int))
print()

print("--------Classification Report--------------")
print()

print(classification_report(yvalid,prediction_int))



y_pred_prob = svc.predict_proba(xvalid_bow)[:,1]

from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(yvalid, y_pred_prob)
# create plot
plt.plot(fpr, tpr, label='ROC curve')
plt.plot([0, 1], [0, 1], 'k--', label='Random guess')
_ = plt.xlabel('False Positive Rate')
_ = plt.ylabel('True Positive Rate')
_ = plt.title('ROC Curve')
_ = plt.xlim([-0.02, 1])
_ = plt.ylim([0, 1.02])
_ = plt.legend(loc="lower right")

plt.show()

from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve 

print()

print("     ROC_AUC_SCORE = ",roc_auc_score(yvalid,y_pred_prob))




precision, recall, thresholds = precision_recall_curve(yvalid, y_pred_prob)
# create plot
plt.plot(precision, recall, label='Precision-recall curve')
_ = plt.xlabel('Precision')
_ = plt.ylabel('Recall')
_ = plt.title('Precision-recall curve')
_ = plt.legend(loc="lower left")

plt.show()




from sklearn.metrics import average_precision_score

print()
print("     Average_Precision_Score = ",average_precision_score(yvalid, y_pred_prob))










