
# KMAMIN 62182275 - EECS 486 A2
# The Vector Space Model
import sys
import re
import nltk
from collections import defaultdict
from text_preprocess import txt_preprocesser
import math
import os
import json

class IRsystem():
    """docstring for IRsystem."""
    def __init__(self, cl_args):
        nltk.download('punkt')
        nltk.download('stopwords')
        # PREPROCESSER
        self.preprocesser = txt_preprocesser()

        # Variables for IR SYS
        self.inv_index = defaultdict(lambda: defaultdict(int))

        self.docvecs = defaultdict(lambda: defaultdict(int)) # store document vectors

        self.docmag = defaultdict(int) # store magnitude of each docvector

        # Begin Processing
        self.args = cl_args[:2] # check again
        self.numdocs = 1400
        # Queries
        with open(cl_args[3]) as queries :
            self.query = queries.read()
            #self.querylist = self.querylist.splitlines()

        #Document by Document
        #flist = os.listdir(cl_args[2])
        #flist.sort()
        #docid = 1
        #for fname in flist:
        with open(cl_args[2]) as file:
            corpus = json.loads(file.read().encode('utf-8'))
            corpus_2 = defaultdict(str)
            for artist,songlist in corpus.items():
                for song in songlist:
                    lyrics = song['lyrics'].strip('\\')
                    corpus_2[artist] += lyrics
                self.indexDocument(corpus_2[artist],artist)
        self.weight_docs()# calculate weights magnitudes

    def weight_docs(self):
        #CHOOSE SCHEMA BASED ON ARGS
        if self.args[0] == 'tfidf':
            self.finish_tfidf()
        else:
            self.finish_tfc()

    def weight_query(self):
        # choose scheme based on ARGS
        if self.args[1] == 'tfidf':
            self.finish_tfidf_query()
        else:
            self.finish_tfc_query()

    def indexDocument(self,fstring,docid):
        """ARGUMENTS notes
        self.args --> [weighting1,weighting2]
        self.inv_index --> inverted index
        fstring - document in string format
        docid - docid number
        NO RETURN VALUE
        OUTPUT --> self.inv_index
        WEIGHTING IS PERFORMED ON LINE 45 at end of __INIT__
        """
        word_list = self.preprocesser.process(fstring)
        #import pdb; pdb.set_trace() # check preprocessed text

        for word in word_list[1]:
            self.inv_index[word][docid] += 1 # add to inverted index
            self.docvecs[docid][word] +=1 # add the TF to docvec

        #print(docid)

    def indexQuery(self,fstring,docid):

        #word_list = re.split(r'\s+',fstring)
        word_list = self.preprocesser.process(fstring)
        #import pdb; pdb.set_trace() # check preprocessed text

        for word in word_list[1]:
            #self.inv_index[word][docid] += 1 # add to inverted index
            self.docvecs[docid][word] +=1 # add the TF to docvec

        return word_list[1]
        #print(docid)

    def finish_tfc(self):
        # calculate the TFCNFX document/term weight
        for docid,dvec in self.docvecs.items() :
            mag = 0
            for key,val in dvec.items():
                dvec[key] = val * math.log10(self.numdocs / len(self.inv_index[key]))
                mag +=  dvec[key] * dvec[key]
            mag = math.sqrt(mag)
            for key,val in dvec.items():
                dvec[key] = val / mag
        # magnitude calculations sum/square
        for docnum,docvec in self.docvecs.items(): # loop through docvecs
            for word, weight in docvec.items():
                self.docmag[docnum] += weight * weight
            self.docmag[docnum] = math.sqrt(self.docmag[docnum])

    def finish_tfc_query(self):
        # calculate the TFCNFX query weight
        maxfq = max(self.docvecs[0].items(),key = lambda x : x[1] )
        for key,val in self.docvecs[0].items():
            self.docvecs[0][key] = ((0.5 + (0.5 * val) / maxfq[1]))  * math.log10(self.numdocs /(1+ len(self.inv_index[key])))
        # magnitude calculation:q
        # assumption of 1 --> to not mess up inverted index * should be robust -
        for word, tfidf in self.docvecs[0].items():
                self.docmag[0] += tfidf * tfidf

        self.docmag[0] = math.sqrt(self.docmag[0])

    def finish_tfidf(self):
        # finish tfidf calcs with tf currently stored in docvecs
        for docid,dvec in self.docvecs.items() :
            for key,val in dvec.items():
                dvec[key] = val * math.log10((self.numdocs) / len(self.inv_index[key]))
        #finish magnitude calculations
        for docnum,docvec in self.docvecs.items(): # loop through docvecs
            for word, tfidf in docvec.items():
                self.docmag[docnum] += tfidf * tfidf
            self.docmag[docnum] = math.sqrt(self.docmag[docnum])


    def finish_tfidf_query(self):
        # finish tfidf calcs with tf currently stored in docvecs
        for key,val in self.docvecs[0].items():
            self.docvecs[0][key] = val * math.log10((self.numdocs) /(1+ len(self.inv_index[key])))
        # finish magnitude calculations
        for word, tfidf in self.docvecs[0].items():
                self.docmag[0] += tfidf * tfidf

        self.docmag[0] = math.sqrt(self.docmag[0])

    def retrieveDocuments(self,query):  #The REQUIRED ARGS ARE INSIDE OF SELF
        """ARGUMENTS notes
        self.args --> [weighting1,weighting2]
        self.inv_index --> inverted index
        query
        """
        # empty the vector and magnitude for new query values
        self.docvecs[0].clear()
        self.docmag[0] = 0

        querywords = self.indexQuery(query,0) # index query and RETURN LIST OF QUERY WORDS
        # THIS FUNCTION IS SIMILAR TO INDEX DOCUMENTS

        """BUILD THE SET OF REL DOCS"""
        queryset = set(querywords)
        doc_set = set()
        for word in queryset:
            if self.inv_index.get(word,False):
                for key,val in self.inv_index[word].items():
                    if key not in doc_set:
                        doc_set.add(key)

        self.weight_query() # BUILD ALL QUERY WEIGHTS --> chooses based of self.args the schema
        score = 0
        ranklist = defaultdict(int)

        """LOOP THROUGH ALL REL DOCS"""
        """SCORE EACH FOR RANKED LIST"""
        """SCORE BASED ON COS SIMILARITY"""
        for docnums in doc_set:
            for index,value in enumerate(self.docvecs[0]):
                score += self.docvecs[0][value] * self.docvecs[docnums][value] # dotproduct
            score = score / (self.docmag[docnums] * self.docmag[0])
            ranklist[docnums] = score
            score = 0

        rl = sorted(ranklist.items(), key=lambda k_v: k_v[1], reverse=True)
        return rl # return SORTED RANKLIST

    def search_all(self): # per single query
        qnum = 1

        # FOR EACH QUERY IN QUERYLIST (which was built in INIT)
        # run retrieveDocuments fxn and PRINT TO FILE

        rl = self.retrieveDocuments(self.query)
        return rl[0][0]

            #print (hitavg / 225 )
            #print(rhitavg / 225)
            #print(rhitavg/225)



def vecspace_main():
    #print(sys.argv)
    args = ["tfidf","tfidf","corpus_data/preprocessedf_corpus.json","output.txt"]
    #["tfidf","tfidf","corpus_data/", # a 1 - 4



    ir_sys = IRsystem(args) # build system
    """ __INIT__ FXN will run INDEX DOCUMENT function """
    output = ir_sys.search_all() # run on ALL QUERIES
    """ will run retrieve documents """
    return output
