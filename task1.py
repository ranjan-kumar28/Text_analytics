import nltk
import pandas as pd
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nltk.tokenize import sent_tokenize,word_tokenize
from nltk.probability import FreqDist
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem import PorterStemmer

#Sentence Tokenization of a text
def Sentence_Tokenization(text):
    return sent_tokenize(text)

#Word Tokenization of a text
def Word_Tokenization(text):
    return word_tokenize(text)

#Frequency Distribution of words in a text
def Frequency_Distribution(list_of_words):
    return FreqDist(list_of_words)

# Frequency Distribution Plot
def Frequency_distribution_Plot(List):
    List.plot(30,cumulative=False)
    plt.show()

#Removing Stopwords
stop_words=set(stopwords.words("english"))
def Removing_Stopwords(List):
    filtered_sent=[]
    for w in List:
        if w not in stop_words:
            filtered_sent.append(w)
    return filtered_sent

'''Lexicon Normalization
Lexicon normalization considers another type of noise in the text.
For example, connection, connected, connecting word reduce to a common word"connect".
It reduces derivationally related forms of a word to a common root word.
'''
def Stemming(List):
    stemmed_words=[]
    for w in List:
        stemmed_words.append(PorterStemmer.stem(w))
    return stemmed_words

'''
Lemmatization reduces words to their base word, which is linguistically correct lemmas.
It transforms root word with the use of vocabulary and morphological analysis.
Lemmatization is usually more sophisticated than stemming.
Stemmer works on an individual word without knowledge of the context.
For example, The word "better" has "good" as its lemma.
This thing will miss by stemming because it requires a dictionary look-up.
'''
def Lemmatization(word):
    return WordNetLemmatizer.lemmatize(word,"v")

'''
POS Tagging
The primary target of Part-of-Speech(POS) tagging is to identify the grammatical group of a given word.
Whether it is a NOUN, PRONOUN, ADJECTIVE, VERB, ADVERBS, etc. based on the context.
POS Tagging looks for relationships within the sentence and assigns a corresponding tag to the word.
'''
def POS_Tagging(tokens):
    return nltk.pos_tag(tokens)

#Till now, you have learned data pre-processing using NLTK. Now, you will learn Text Classification. 
#############################################################
###Performing Sentiment Analysis using Text Classification###
#############################################################
#use of vaderSentiment module for sentiment analysis

'''
About the Scoring
The compound score is computed by summing the valence scores of each word in the lexicon,
adjusted according to the rules, and then normalized to be between -1 (most extreme negative) and +1 (most extreme positive).
This is the most useful metric if you want a single unidimensional measure of sentiment for a given sentence.
Calling it a 'normalized, weighted composite score' is accurate.
It is also useful for researchers who would like to set standardized thresholds for classifying sentences as either positive, neutral, or negative.
Typical threshold values (used in the literature cited on this page) are:

positive sentiment: compound score >= 0.05
neutral sentiment: (compound score > -0.05) and (compound score < 0.05)
negative sentiment: compound score <= -0.05
'''
analyser = SentimentIntensityAnalyzer()
def sentiment_analyzer_scores(sentence):
    score = analyser.polarity_scores(sentence)
    return score
def sentiment_analyzer_scores1(sentence):
    score = analyser.polarity_scores(Removing_Stopwords(Word_Tokenization(sentence)))
    List = list()
    List.append(['Compound',str(score['compound'])])
    List.append(['Positive',int(score['pos'])])
    List.append(['Negative',int(score['neg']))
    List.append(['Neutral',int(score['neu']))
    return List

def create_excel_sentiment_results(df):
    Combined_List=list()
    for index in df.index:
        individual_list=list()
        individual_list.append(df['PhraseId'][index])
        individual_list.append(df['Phrase'][index])

        score=sentiment_analyzer_scores(str(df['Phrase'][index]))

        individual_list.append(score['neg'])
        individual_list.append(score['pos'])
        individual_list.append(score['neu'])
        individual_list.append(score['compound'])
        Combined_List.append(individual_list)
        del individual_list
    new_df=pd.DataFrame(Combined_List,columns=['PhraseId','Phrase','Negative sentiment','Positive sentiment','Neutral score','Compound score'])
    new_df.to_excel('results.xlsx')

def plot_sentiments(df):
    names=['positive','negative']
    size=list()
    size.append(df['Positive sentiment'].sum())
    size.append(df['Negative sentiment'].sum())
    my_circle=plt.Circle( (0,0), 0.7, color='white')
    plt.pie(size, labels=names, autopct='%1.1f%%', shadow=True, colors=['#edcf3b','#c65959'])
    p=plt.gcf()
    p.gca().add_artist(my_circle)
    plt.show()

def main():
    df=pd.read_excel('test.xlsx')
    create_excel_sentiment_results(df)
    plot_sentiments(pd.read_excel('results.xlsx'))

#check if it's directly run or imported
#if not imported then call main function
#if __name__ == '__main__':
#	main()
