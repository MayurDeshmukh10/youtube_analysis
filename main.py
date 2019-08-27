#all basic features are working according to kernel

import pandas as pd
import numpy as np
import seaborn as sns
import json
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
from textblob import TextBlob

videos = pd.read_csv('USvideos.csv',encoding='utf8',error_bad_lines = False)#opening the file USvideos
comm = pd.read_csv('UScomments.csv',encoding='utf8',error_bad_lines=False) #opening the file UScomments

print(videos)

pol=[] # list which will contain the polarity of the comments
for i in comm.comment_text.values:
    try:
        analysis =TextBlob(i)
        pol.append(analysis.sentiment.polarity)

    except:
        pol.append(0)


comm['pol']=pol

comm['pol'][comm.pol==0]= 0

comm['pol'][comm.pol > 0]= 1
comm['pol'][comm.pol < 0]= -1


df_positive = comm[comm.pol==1]
df_positive.head()


k= (' '.join(df_positive['comment_text']))

'''wordcloud = WordCloud(width = 1000, height = 500).generate(k)
plt.figure(figsize=(15,5))
plt.show(wordcloud)
plt.axis('off')'''


'''df_negative = comm[comm.pol==-1]
k= (' '.join(df_negative['comment_text']))
wordcloud = WordCloud(width = 1000, height = 500).generate(k)
plt.figure(figsize=(15,5))
plt.imshow(wordcloud)
plt.axis('off')'''


comm['pol'].replace({1:'positive',0:'Neutral',-1:'negative'}).value_counts().plot(kind='bar',figsize=(7,4))
plt.title('Number of types of commets')
plt.xlabel('Comment_type')
plt.ylabel('number')
plt.show()

id=[]
pos_comm=[]
neg_comm=[]
neutral_comm =[]
for i in set(comm.video_id):
    id.append(i)
    try:
        pos_comm.append(comm[comm.video_id==i].pol.value_counts()[1])
    except:
        pos_comm.append(0)
    try:
        neg_comm.append(comm[comm.video_id==i].pol.value_counts()[-1])
    except:
        neg_comm.append(0)
    try:
        neutral_comm.append(comm[comm.video_id==i].pol.value_counts()[0])
    except:
        neutral_comm.append(0)


df_unique = pd.DataFrame(id)
df_unique.columns=['id']
df_unique['pos_comm'] =pos_comm
df_unique['neg_comm'] = neg_comm
df_unique['neutral_comm'] = neutral_comm
df_unique['total_comments']=df_unique['pos_comm']+df_unique['neg_comm']+df_unique['neutral_comm']
df_unique.head(6)


df_unique.to_csv('unique.csv',index=False,)

videos.head()

comm.head()

videos.date.value_counts()

print(videos.video_id.value_counts()[:12]) # these videos have become 7 times the most trending videos of these 2 weeks.
most_trending = videos.video_id.value_counts()[:12].index

videos[videos.video_id=='mlxdnyfkWKQ']

for i in most_trending:
    info =videos[videos.video_id== i][['title','channel_title','views','likes','dislikes','comment_total']].tail(1)# get the last row of the dataframe(total like,views,dislikes)
    print(info)
    print('****************************************************************************************')

# slpitting the tags
tags = videos['tags'].map(lambda k: k.lower().split('|')).values

# joining and making a complete list
'''k= (' '.join(videos['tags']))
wordcloud = WordCloud(width = 1000, height = 500).generate((' '.join(k.lower().split('|'))))# word cloud


plt.figure(figsize=(15,5))
plt.imshow(wordcloud)
plt.axis('off')'''

videos.columns

df1 =pd.DataFrame(videos.channel_title.value_counts())
df1.columns=['times channel got trenidng']# how many times the channel got trending'
df1.head(6)

df_channel =pd.DataFrame(videos.groupby(by=['channel_title'])['views'].mean()).sort_values(by='views',ascending=False)
df_channel.head(10).plot(kind='bar')
plt.title('Most viewed channels')
plt.show()


df_channel =pd.DataFrame(videos.groupby(by=['channel_title'])['likes'].mean()).sort_values(by='likes',ascending=False)
df_channel.head(10).plot(kind='bar')
plt.title('Most liked channels')
plt.show()

videos['likes_per_view']=videos['likes']/videos['views']
df_channel =pd.DataFrame(videos.groupby(by=['channel_title'])['likes_per_view'].mean()).sort_values(by='likes_per_view',ascending=False)
df_channel.head(10).plot(kind='bar')
plt.title('Most liked channels')
plt.show()

videos['dislikes_per_view']=videos['dislikes']/videos['views']
df_channel =pd.DataFrame(videos.groupby(by=['channel_title'])['dislikes_per_view'].mean()).sort_values(by='dislikes_per_view',ascending=False)
df_channel.head(10).plot(kind='bar')
plt.title('Most disliked channels')
plt.show()


unique = pd.read_csv('unique.csv',)

unique.sort_values(by='pos_comm',ascending=False).head(5)

videos[videos.video_id == 'eERPlIdPJtI'].title[225]


sns.barplot(data=unique.sort_values(by='pos_comm',ascending=False).head(10),x='id',y='pos_comm')
plt.xticks(rotation=45)
plt.figure(figsize=(5,4))
plt.show()

sns.barplot(data=unique.sort_values(by='neg_comm',ascending=False).head(10),x='id',y='neg_comm')
plt.xticks(rotation=45)
plt.figure(figsize=(5,4))
plt.show()

sns.barplot(data=unique.sort_values(by='total_comments',ascending=False).head(10),x='id',y='total_comments')
plt.xticks(rotation=45)
plt.figure(figsize=(5,4))
plt.show()

sns.regplot(data=videos,x='views',y='likes')
plt.title("Regression plot for likes & views")
plt.show()

sns.regplot(data=videos,x='views',y='dislikes')
plt.title("Regression plot for dislikes & views")
plt.show()


df_corr = videos[['views','likes','dislikes']]

sns.heatmap(df_corr.corr(),annot=True)
plt.show()
