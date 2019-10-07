import os
import re
import sys
import nltk
from nltk.corpus import stopwords
from collections import defaultdict
# KMAMIN KMAMIN ---62182275 Krishan Amin

# MODIFIED A1 code
class txt_preprocesser():
    # Various Constants  - in sets for later use
    date_dict = {'january','february','march','april',',may','june','july',
                'august','september','october','november','december'}
    day_dict = {'monday','tuesday','wednesday','thursday','friday','saturday','sunday'}
    abbrev = {'mr','ms','mrs','dr','vs','etc',''}
    pronouns = {'he','she','it','i','who','what','where','when','this','that'}
    con_exp = {'s':'is','m':'am','re':'are','ll':'will'}
    skip = {'','.',',',' .','. ',' . '}

    def __init__(self):

        self.total_words = 0
        self.vocab_size = 0
        #nltk.download('stopwords')
        #nltk.download('wordnet')
        self.stop_dict = set(stopwords.words('english'))
        #self.word_freq = defaultdict(int) # store word - freq pairs
        # initialize class variables

    def removeSGML(self,text):
        # use RE to remove any <TEXT> in this form
        # re performed on ENTIRE docstring
         text = re.sub(r'<.*?>','',text)
         return text

    def removeStopwords(self,wordList):
        wordList[:] = [word for word in wordList if word not in self.stop_dict]
        # LIST COMP to remove any words that are also found in the dictionary of stopwords
        # new list returned has NO STOPWORDS
        return wordList

    def tokenize(self,text):
        text = text.lower()
        text = re.sub(r',','',text)
        text = re.sub(r'/','',text)
        list = nltk.word_tokenize(text)
        # docid = list[0]
        index = 0
        while(index < len(list)):

            if list[index] in self.skip: # SKIP PUNCTUATION & USELESS CHARS
                del list[index]
                continue
            if list[index] in self.date_dict: # IF DATE  / WORD FOUND
                if list[index+1].isdigit(): # check if following digit
                    list[index] = " ".join([list[index],list[index+1]])
                    del list[index+1]
                    if list[index+1].isdigit(): # check for year
                        list[index] = " ".join([list[index],list[index+1]])
                        del list[index+1]
                index += 1 # increment index in list
                continue
            if list[index] in self.day_dict: # days DATE check
                if list[index+1].isdigit(): # check if following digit
                    list[index] = " ".join([list[index],list[index+1]])
                    del list[index+1]
                index += 1 # increment index in list
                continue
            if(list[index][0].isdigit()): # REMOVE nums - check if first char is a num
                # if the first char is number then make sure the rest of number is, or just continue
                if(list[index].isdigit()): #check if its not a date
                    del list[index]
                    continue
            if re.search(r'\.',list[index]): #search for periods
                list[index] = list[index][:-1]
                if re.search(r'\.',list[index]): #check if > 1 period
                    del list[index]
                    continue
                elif (list[index][0].isdigit()): #delete (prev) decimals
                    del list[index]
                    continue
                elif list[index] in self.abbrev: #delete abbreviations
                    del list[index]
                    continue
                elif len(list[index]) < 3: #delete very short terms (probably abbrev)
                    del list[index]

            if re.search(r"'",list[index]): #apostrophe
                #import pdb; pdb.set_trace()
                wl = re.split(r"'",list[index]) # split on apostrophe
                list[index] = wl[0]
                if wl[0] in self.pronouns:  # check first term if "PRONOUN"
                    list.insert(index+1,self.con_exp.get(wl[1],"'"+wl[1])) # expand
                else :
                    list.insert(index+1,"'"+wl[1]) # if not pronoun then add original term after apostrophe
                index += 2 # increment and account for the new term
                continue
            index += 1
            
        return list

    def stemWords(self,wordList):
         stemmer = nltk.PorterStemmer() # use NLTK PorterStemmer to STEM
         i = 0
         while i < len(wordList):
             wordList[i] =  stemmer.stem(wordList[i])
             i += 1
         return wordList


    def process(self,text):
        #print(fname+' begins -->', end='')

        #process text in the 4 req steps
        # text = self.removeSGML(text) # string --> list
        listwdocid = self.tokenize(text)
        #listwdocid = self.removeStopwords(listwdocid)
        # listwdocid = self.stemWords(listwdocid)
        # UPDATE CLASS VARIABLES [total words & frequency dict]
        #self.total_words += len(list)
        #for word in list:
        #    self.word_freq[word] += 1
        #print('completed')

        return listwdocid
