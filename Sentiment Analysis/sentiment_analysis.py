import re
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
import string
from termcolor import colored, cprint
import nltk
import warnings 
warnings.filterwarnings("ignore")


text1 = colored('Step 1 : Loading Training and Test Dataset','red',attrs=['bold'])
print("\n\n")
print(text1)
print('\n\n')

f = input()


train = pd.read_csv("train.csv")

text2 = colored('  Summary of Training data ','green',attrs=['bold'])

print(text2)
print("\n")

print(train.describe(include='all'))

test = pd.read_csv("test.csv")

print("\n\n")

text2 = colored('  Summary of Testing data ','green',attrs=['bold'])

print(text2)
print("\n")

print(train.describe(include='all'))


test = pd.read_csv("test.csv")

f = input()

print("\n\n")
text1 = colored('Step 2 : Preprocessing of Data','red',attrs=['bold'])
print("\n\n")
print(text1)
print("\n")

text3 = colored("2.1 Removal of NA Values",'green',attrs=['bold'])

f = input()


print(text3)
print("\n\n")
text2 = colored('\tNo. of NA Values before preprocessing','blue',attrs=['bold'])

print(text2)
print("\tTotal NA Values : ",train.isnull().sum().sum())
train['Comment']=train['Comment'].fillna("")
test['Comment']=test['Comment'].fillna("")

f = input()

text2 = colored('\tNo. of NA Values After preprocessing','blue',attrs=['bold'])
print("\n")
print(text2)
print("\tTotal NA Values : ",train.isnull().sum().sum())

f = input()


#print(train.head())
print("\n\n")
text3 = colored("2.2 Removing @mentions,Punctuations,Numbers and Special Characters ",'green',attrs=['bold'])

f = input()

print(text3)
print("\n\n")
combi = train.append(test, ignore_index=True)

combi['Comment'] = combi['Comment'].str.replace("[^a-zA-Z#]", " ")

print(combi['Comment'])

f = input()

print("\n\n")
text3 = colored("2.3 Removing Short Words length less than 2 ",'green',attrs=['bold'])
print(text3)
f = input()
print("\n\n")

combi['Comment'] = combi['Comment'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))

print(combi['Comment'])


tokenized_comment = combi['Comment'].apply(lambda x: x.split())

#tokenized_comment.head()

f = input()

print("\n\n")
text3 = colored("2.4 Applying Stemming on Comments ",'green',attrs=['bold'])
print(text3)
f = input()
print("\n\n")

from nltk.stem.porter import *
stemmer = PorterStemmer()

tokenized_comment = tokenized_comment.apply(lambda x: [stemmer.stem(i) for i in x]) # stemming
tokenized_comment.head()

for i in range(len(tokenized_comment)):
    tokenized_comment[i] = ' '.join(tokenized_comment[i])


combi['Comment'] = tokenized_comment

print(combi['Comment'])

f = input()

all_words = ' '.join([text for text in combi['Comment']])


print("\n\n")
text3 = colored("2.4 Displaying Word Clouds ",'green',attrs=['bold'])
print(text3)
print("\n\n")

from wordcloud import WordCloud

normal_words =' '.join([text for text in combi['Comment'][combi['polarity'] == 1]])

wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(normal_words)
plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.title('Positive Words ')
plt.axis('off')
plt.show()



negative_words = ' '.join([text for text in combi['Comment'][combi['polarity'] == 0]])
wordcloud = WordCloud(width=800, height=500,
random_state=21, max_font_size=110).generate(negative_words)
plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.title('Negative Words')
plt.axis('off')
plt.show()


print("\n\n")
text3 = colored("3. Feature Extraction from cleaned comment using Bag of Words ",'red',attrs=['bold'])
print(text3)
f = input()
print("\n\n")

from sklearn.feature_extraction.text import CountVectorizer
bow_vectorizer = CountVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english')
# bag-of-words feature matrix
bow = bow_vectorizer.fit_transform(combi['Comment'])

print(bow)

f = input()

print("\n\n")
text3 = colored("4. Dividing Data into Training and Test Sets ",'red',attrs=['bold'])
print(text3)
f = input()
print("\n\n")

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

train_bow = bow[:17741,:]
test_bow = bow[17742:,:]


# splitting data into training and validation set
xtrain_bow, xvalid_bow, ytrain, yvalid = train_test_split(train_bow, train['polarity'], random_state=42, test_size=0.3)
#xtrain_bow, xvalid_bow, ytrain, yvalid = train_test_split(train_tfidf, train['polarity'], random_state=42, test_size=0.3)

print("\n\n")
text3 = colored("4.1 Training Dataset ",'green',attrs=['bold'])
print(text3)
f = input()
print("\n\n")
print(xtrain_bow)
f = input()


print("\n\n")
text3 = colored("4.2 Test Dataset ",'green',attrs=['bold'])
print(text3)
f = input()
print("\n\n")
print(ytrain)
f = input()


#----------------------------------------------------------------------------------------------------

print("\n\n")
text3 = colored("5. Building a SVM Model.. ",'red',attrs=['bold'])
print(text3)
f = input()
print("\n\n")
print("Building....")
print("\n\n")

from sklearn import svm

svc = svm.SVC(kernel='linear', C=1, probability=True,decision_function_shape='ovo').fit(xtrain_bow, ytrain)


print(svc)

print("\n\n")
text3 = colored("6. Displaying the learning Curve ",'red',attrs=['bold'])
print(text3)

print("\n\n")

from mlxtend.plotting import plot_learning_curves

plot_learning_curves(xtrain_bow,ytrain,xvalid_bow,yvalid,svc)
plt.show()


prediction = svc.predict_proba(xvalid_bow)
prediction_int = prediction[:,1] >= 0.3
prediction_int = prediction_int.astype(np.int)



print("\n\n")
text3 = colored("7. Displaying the Predicted Sentiment Analysis ",'red',attrs=['bold'])
print(text3)
f = input()
print("\n\n")
positive_comments = []
negative_comments = []

for i in prediction_int:
        if i == 0:
            negative_comments.append(i)
        else:
            positive_comments.append(i)



print("\tTotal Predicted Positive Comments : ",len(positive_comments))
print("\n\tTOTAL Predicted Negative Comments : ",len(negative_comments))



plt.bar(["Positive"],[len(positive_comments)],label = "Positive")
plt.bar(["Negative"],[len(negative_comments)],label = "Negative")
plt.legend()
plt.xlabel('Type of Comment')
plt.ylabel('Count of Comment')
plt.title('Predicted Comment Count in Test Dataset')

plt.show()




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

print("\n\n")
text3 = colored("8. F1 Score of Classification ",'red',attrs=['bold'])
print(text3)
f = input()
print("\n\n")

text2 = colored('\tF1 Score : ','blue',attrs=['bold'])

print(text2,f1_score(yvalid, prediction_int))
f = input()

#print("     F1 Score = ",f1_score(yvalid, prediction_int)) # calculating f1 score
print()

print("\n\n")
text3 = colored("9. Displaying Confusion Matrix of Classification ",'red',attrs=['bold'])
print(text3)
f = input()
print("\n\n")

#print("     Confusion Matrix of Model")
#rint()
#print(confusion_matrix(yvalid,prediction_int))
#print()


from mlxtend.plotting import plot_confusion_matrix

confusion_mat = confusion_matrix(yvalid,prediction_int)

class_names = ['Positive','Negative']


fig, ax = plot_confusion_matrix(conf_mat=confusion_mat)
plt.show()


print("\n\n")
text3 = colored("10. Classification Report for Trained SVM Model ",'red',attrs=['bold'])
print(text3)
f = input()
print("\n\n")

print("--------Classification Report--------------")
print()

print(classification_report(yvalid,prediction_int))

f = input()



y_pred_prob = svc.predict_proba(xvalid_bow)[:,1]


print("\n\n")
text3 = colored("11. Area Under Receiver Operating Characteristic Curve(ROC) Score  ",'red',attrs=['bold'])
print(text3)
f = input()
print("\n\n")


from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve 

text2 = colored('\t AUC ROC Score : ','blue',attrs=['bold'])

print(text2,roc_auc_score(yvalid,y_pred_prob))

f = input()

#print("     ROC_AUC_SCORE = ",roc_auc_score(yvalid,y_pred_prob))




print("\n\n")
text3 = colored("12. Average Precision Score  ",'red',attrs=['bold'])
print(text3)
f = input()
print("\n\n")


from sklearn.metrics import average_precision_score

text2 = colored('\t Average Precision Score : ','blue',attrs=['bold'])

print(text2,average_precision_score(yvalid, y_pred_prob))

f = input()

#print("     Average_Precision_Score = ",average_precision_score(yvalid, y_pred_prob))


from sklearn.metrics import accuracy_score

print("\n\n")
text3 = colored("13. Accuracy Score of Trained SVM Model ",'red',attrs=['bold'])
print(text3)
f = input()
print("\n\n")


acc_score = accuracy_score(yvalid,prediction_int)

text2 = colored('\t Accuracy Score : ','blue',attrs=['bold'])

print(text2,acc_score)

f = input()


#print("     Accuracy score = ",acc_score)

print("\n\n")
text3 = colored("14. Brier Loss Score ",'red',attrs=['bold'])
print(text3)
f = input()
print("\n\n")

from sklearn.metrics import brier_score_loss

text2 = colored('\t Brier Loss Score : ','blue',attrs=['bold'])

print(text2,brier_score_loss(yvalid,prediction_int))

f = input()

print("\n\n")
text3 = colored("15. Displaying ROC Curve ",'red',attrs=['bold'])
print(text3)
f = input()
print("\n\n")

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




print("\n\n")
text3 = colored("16. Displaying Precision Recall Curve ",'red',attrs=['bold'])
print(text3)
f = input()
print("\n\n")


precision, recall, thresholds = precision_recall_curve(yvalid, y_pred_prob)
# create plot
plt.plot(precision, recall, label='Precision-recall curve')
_ = plt.xlabel('Precision')
_ = plt.ylabel('Recall')
_ = plt.title('Precision-recall curve')
_ = plt.legend(loc="lower left")

plt.show()









