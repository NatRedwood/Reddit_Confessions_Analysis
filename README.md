# Reddit Confessions Analysis - Text Data Engineering

The analysis focuses on retrieving main topics of the reddit users' confessions and the use of personal pronouns in the confessions. 

## Project Description
The purpose of this project is:
1. Conducting text mining on reddit confessions dataset.
2. Cleaning the data: removing punctuation and non-alphabetic characters.
3. Extracting information: analyzing length of reddits, frequency distribution of vocabulary.
4. Extracting features: distribution of personal pronouns, positive lexicon and negative lexicon.
5. Retrieving main topics: WorcCloud package.

### Methods Used
* Text Mining
* Data Visualization
* Frequency Distribution
* NLP
* IE (Information Extraction)
* IR (Information Retrieval)

### Technologies
* Python, jupyter
* Numpy, Pandas, MatPlotLib
* NLTK, Regexp, WordCloud

## Needs of this project

- text data mining
- data processing/cleaning
- information extraction

## Methodology
To clean the data, get basic frequencies and plot figures showing the distribution of reddits' length, use of personal pronouns, positive lexicon, and negative lexicon, I used Python code. To present the information about the main topics and better visualize the distributions, I used the WordCloud package in Jupyter Notebook.

## Data used
- One Million Reddit Confessions [link](https://www.kaggle.com/pavellexyr/one-million-reddit-confessions) / Data found on Kaggle

## Project files
1. Reddit Confessions Analysis **PY**: [link](https://github.com/Nwojarnik/Reddit_Confessions_Analysis/blob/main/Reddit_confessions._Analysis%2Bcleaning.py)
2. WordClouds Reddit Confessions **JUPYTER**: [link](https://github.com/Nwojarnik/Reddit_Confessions_Analysis/blob/main/WordClouds%20JUPYTER.ipynb)
3. Twister Tongue Corpus Analysis **PY**: [link](https://github.com/Nwojarnik/Twister_Tongue_Corpus_Analysis/blob/main/Twister_Tongue_Analysis%20PYTHON.py)

Files used for creating the wordclouds:
4. List of words after cleaning **TXT**: [link](https://github.com/Nwojarnik/Reddit_Confessions_Analysis/blob/main/clean_text.txt)
5. Freq Dist of words **CSV**: [link](https://github.com/Nwojarnik/Reddit_Confessions_Analysis/blob/main/freq_dist_clean.csv)
6. Freq Dist of positive lexicon **CSV**: [link](https://github.com/Nwojarnik/Reddit_Confessions_Analysis/blob/main/freq_dist_pos.csv)
7. Freq Dist of negative lexicon **CSV**: [link](https://github.com/Nwojarnik/Reddit_Confessions_Analysis/blob/main/freq_dist_neg.csv)
8. Freq Dist of pronouns **CSV**: [link](https://github.com/Nwojarnik/Reddit_Confessions_Analysis/blob/main/freq_dist_pronouns.csv)
9. Freq Dist of 20 first most common words **CSV**: [link](https://github.com/Nwojarnik/Reddit_Confessions_Analysis/blob/main/most_common_20.csv)

## Project plots and graphs
1. Distribution of length: [link](https://github.com/Nwojarnik/Reddit_Confessions_Analysis/blob/main/dist_of_length.png)
2. Distribution of pronouns: [link](https://github.com/Nwojarnik/Reddit_Confessions_Analysis/blob/main/dist_pronouns.png)
3. Distribution of personal pronouns SG vs PL: [link](https://github.com/Nwojarnik/Reddit_Confessions_Analysis/blob/main/dist_pronouns_sg_pl_comparison.png)
4. Distribution of positive vs negative words: [link](https://github.com/Nwojarnik/Reddit_Confessions_Analysis/blob/main/pos_neg_lexicon.png)
5. Positive words distribution: [link](https://github.com/Nwojarnik/Reddit_Confessions_Analysis/blob/main/dist_pos_words.png)
6. Negative words distribution: [link](https://github.com/Nwojarnik/Reddit_Confessions_Analysis/blob/main/dist_neg_words.png)

## Project wordclouds
1. Main topics wordcloud: [link](https://github.com/Nwojarnik/Reddit_Confessions_Analysis/blob/main/wordcloud_clean_text.png)
2. Wordcloud from distribution: [link](https://github.com/Nwojarnik/Reddit_Confessions_Analysis/blob/main/wordcloud_freq_dist_clean.png)
3. Most common words wordcloud: [link](https://github.com/Nwojarnik/Reddit_Confessions_Analysis/blob/main/wordcloud_most_common.png)
4. Pronouns wordcloud: [link](https://github.com/Nwojarnik/Reddit_Confessions_Analysis/blob/main/wordcloud_pronouns.png)
5. Positive words wordcloud: [link](https://github.com/Nwojarnik/Reddit_Confessions_Analysis/blob/main/wordcloud_positive_words.png)
6. Negative words wordcloud: [link](https://github.com/Nwojarnik/Reddit_Confessions_Analysis/blob/main/wordcloud_negative_words.png)
