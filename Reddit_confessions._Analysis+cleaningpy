import re
import nltk
from nltk.probability import FreqDist
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


'''Reading the data from Kaggle. Dropping any other columns
except text in itself since that's the focus of the analysis.
Reading only first 1000 reddits.'''


df = pd.read_csv('C:/Users/natal/PROJECT REDDIT CONFESSIONS/one-million-reddit-confessions.csv', encoding = 'utf-8')
df = df.iloc[0:1000,10]
print(df)


'''Basic plotting and counting to see the distribution 
of length over the reddits.'''


def length_reddits(x):
    lst = []
    for i in x:
        lst.append(len(i))
    return lst

length = np.array(length_reddits(df))
type(df)
df = df.to_frame()
type(df)
df.insert(1, column = 'length', value = length)
df.head()

#dropping null values if exist

print(df.shape)
print(df.isnull().values.any())
df.dropna(axis = 0, inplace = True)
print(df.shape)
df
 
#simple plot of length counts of reddits

values = df['length']
plt.plot(values)
plt.ylabel('Length Count')
plt.xlabel('Reddits Indexes')
plt.title("Distribution of length")
plt.show()


'''Splitting the reddits into words and looking at the 
use of pronouns + negative and positive words'''


def split_reddits(x):
    x = [i.split() for i in x]
    all = []
    for sublist in x:
        for item in sublist:
            all.append(item)
    return all

def lower_reddits(x):
    x = [i.lower() for i in x]
    return x

#removing special characters like ' in i'm and it's

def remove_spec_char(x):
    all = []
    for i in x:
        i = re.sub('[^a-zA-Z\s]', ' ', i)
        all.append(i)
    return all

#removing single characters from the abbreviations
#(s, d, m)

def remove_single_char(x):
    all = []
    for i in x:
        if i in ['s','d','m','t','re']:
            continue
        all.append(i)
    return all

def clean_reddits(x):
    _steps = [lower_reddits, remove_spec_char, split_reddits, remove_single_char]

    for step in _steps:
        x=step(x)
    return x

clean_text = clean_reddits(df['title'])
clean_text[100:300]

#Frequency Distribution in confessions and vocabulary analysis

tokens_count = len(clean_text)
tokens_count
clean_text_types = sorted(list(set(clean_text)))
types_count = len(clean_text_types)
types_count

first_sg_pronouns = ['i', 'me','my','mine','myself']
second_sg_pronouns = ['you', 'your','yours','yourself']
third_sg_pronouns = ['he', 'she','it', 'her', 'his','her','himself','herself','itself']
first_pl_pronouns = ['we','us','our','ours','ourselves']
second_pl_pronouns = ['you', 'your','yours','yourselves']
third_pl_pronouns = ['they','them','their','theirs','themselves']
all_personal_pronouns = set(first_sg_pronouns + second_sg_pronouns + third_sg_pronouns + first_pl_pronouns + second_pl_pronouns + third_pl_pronouns)
all_personal_pronouns

def percentage(x,y,base,target):
    z = (x/y) * 100
    return print(f'Percentage of {target} in {base}:{ round(z,2)} %')

def count_pronouns(text, pronouns):
    count = 0
    for word in text:
        if word in pronouns:
            count = count + 1
        else:
            continue
    return count

pronouns_count = count_pronouns(clean_text, all_personal_pronouns)
pronouns_count

percentage(types_count, tokens_count, 'types', 'tokens')
percentage(pronouns_count, len(clean_text), 'personal pronouns', 'tokens')



freq_dist_text = FreqDist(clean_text)
df_freq_dist_text = pd.DataFrame(list(freq_dist_text.items()), columns = ['Word','Count'])
df_freq_dist_text

most_common = freq_dist_text.most_common(20)
most_common = pd.DataFrame(data = freq_dist_text.most_common(20), columns = ['Word', 'Count'])
most_common

freq_dist_text.plot(20,cumulative = True)

clean_text_types[:25]
all_personal_pronouns
pronouns_reddits = sorted([w for w in clean_text if w in all_personal_pronouns])
sorted(set(pronouns_reddits))

freq_dist_pronouns = FreqDist(pronouns_reddits)
freq_dist_pronouns = pd.DataFrame(list(freq_dist_pronouns.items()), columns = ['Pronoun', 'Count'])
freq_dist_pronouns
list(freq_dist_pronouns.items())

freq_dist_pronouns.plot.bar(x = 'Pronoun', y='Count', rot=80, title='Count of pronouns in confession reddits')
plt.show()

#Plotting and analyzing different types of pronouns

first_sg_pronouns = ['i', 'me','my','mine','myself']
second_sg_pronouns = ['you', 'your','yours','yourself']
third_sg_pronouns = ['he', 'she','it', 'her', 'his','her','himself','herself','itself']
first_pl_pronouns = ['we','us','our','ours','ourselves']
second_pl_pronouns = ['you', 'your','yours','yourselves']
third_pl_pronouns = ['they','them','their','theirs','themselves']
all_personal_pronouns = set(first_sg_pronouns + second_sg_pronouns + third_sg_pronouns + first_pl_pronouns + second_pl_pronouns + third_pl_pronouns)
all_personal_pronouns

data = {'Person': ['1st_person', '2nd_person', '3rd_person'], 
'SG': [count_pronouns(clean_text, first_sg_pronouns), count_pronouns(clean_text, second_sg_pronouns), count_pronouns(clean_text, third_sg_pronouns)],
'PL': [count_pronouns(clean_text, first_pl_pronouns),count_pronouns(clean_text, second_pl_pronouns), count_pronouns(clean_text, third_pl_pronouns)]}

df_pronouns_person = pd.DataFrame(data)
df_pronouns_person

df_pronouns_person.plot.bar(x='Person', rot=0, title='Distribution of personal pronouns in 1st, 2nd and 3rd person SG/PL\nin confession reddits', xlabel = 'Person', ylabel='Counts', fontsize = 'large', color = ['orange', 'yellow'])
plt.text(0.5,1000, df_pronouns_person.to_string(index=False))
plt.legend(labels=['Singular', 'Plural'])
plt.show()

f3 = open('C:/Users/natal/CU BOULDER NLP COURSE/Assignment2/positive-words.txt', encoding='utf8')
f4 = open('C:/Users/natal/CU BOULDER NLP COURSE/Assignment2/negative-words.txt', encoding='utf8')
lex_pos = f3.read()
lex_neg = f4.read()
lex_pos = lex_pos.split()
lex_neg = lex_neg.split()

def count_pos(lst):
    pos_results = []
    for word in lst:
        if word in lex_pos:
            pos_results.append(word)
    return len(pos_results)

def count_neg(lst):
    neg_count = 0
    for word in lst:
        if word in lex_neg:
            neg_count = neg_count + 1
    return neg_count

#Pos vs neg word count compared

x = ['Positive lexicon in confessions', 'Negative lexicon in cofessions']
y = [count_pos(clean_text), count_neg(clean_text)]
plt.bar(np.arange(len(x)), y, color = 'cyan')
plt.xticks(np.arange(len(x)), ['Positive lexicon in confessions', 'Negative lexicon in cofessions'])
plt.ylabel('Counts')
plt.title('Positive vs negative words')
plt.text(0.00001, 800, f'Positive words: {y[0]}\nNegative words: {y[1]}')
plt.show()

pos_lex_reddits = sorted([w for w in clean_text if w in lex_pos and clean_text.count(w) > 2])
sorted(set(pos_lex_reddits))

freq_dist_pos = FreqDist(pos_lex_reddits)
freq_dist_pos
freq_dist_pos = pd.DataFrame(list(freq_dist_pos.items()), columns = ['Positive Word', 'Count'])
freq_dist_pos

freq_dist_pos.plot.scatter(x="Positive Word", y = 'Count', color='red', rot=85)
plt.show()

neg_lex_reddits = sorted([w for w in clean_text if w in lex_neg and clean_text.count(w) > 5])
sorted(set(neg_lex_reddits))

freq_dist_neg = FreqDist(neg_lex_reddits)
freq_dist_neg
freq_dist_neg = pd.DataFrame(list(freq_dist_neg.items()), columns = ['Negative Word', 'Count'])
freq_dist_neg

freq_dist_neg.plot.scatter(x="Negative Word", y = 'Count', color='red', rot=85)
plt.show()


'''WORD COULD VISUALIZATION

Visualizations will be done in another Jupyter Notebook file

Preapering the data for wordclouds to export and read in a separate notebook'''


#exporting clean text as txt/ first 1000 reddits as text,cleaned

clean_text[:100]
with open("C:/Users/natal/PROJECT REDDIT CONFESSIONS/clean_text.txt", "w") as output:
    output.write(str(clean_text))

#exporting words and counts (freq dist) from cleaned reddit as csv
df_freq_dist_text.to_csv("C:/Users/natal/PROJECT REDDIT CONFESSIONS/freq_dist_clean.csv")

#exporting most common 20 words ans counts as csv
most_common.to_csv("C:/Users/natal/PROJECT REDDIT CONFESSIONS/most_common_20.csv")

#exporting freq dist of pronouns in reddits
freq_dist_pronouns.to_csv("C:/Users/natal/PROJECT REDDIT CONFESSIONS/freq_dist_pronouns.csv")

#exporting freq dist of pos words in reddits
freq_dist_pos.to_csv("C:/Users/natal/PROJECT REDDIT CONFESSIONS/freq_dist_pos.csv")

#exporting freq dist of neg words in reddits
freq_dist_neg.to_csv("C:/Users/natal/PROJECT REDDIT CONFESSIONS/freq_dist_neg.csv")
