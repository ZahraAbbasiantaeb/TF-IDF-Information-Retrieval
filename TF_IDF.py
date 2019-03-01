from nltk.tokenize import word_tokenize
import nltk as nltk
import numpy as np
import pandas as pd

# this part defines path of data files
path = 'Hamshahri-Corpus.txt'

queryFilePath = 'Querys.txt'

judgmentPath = 'judgment.txt'

pathToWriteResults = 'result.txt'


# this function is used to parse Querys file. given the name of tag it returns all the data within it
def parseXML(path,tag):
    
    begin_str = '<'+tag+'>'
    
    end_str = '</'+tag+'>'
    
    token = ""
    
    file = open(path,'r')
    
    begin = 0
    
    values = []
    
    for line in file:

        match = line.find(begin_str)
        
        if match == 0:
            begin = 1

        match2 = line.find(end_str)
        
        if match2 == 0:
            
            begin = 0
            
            token = token.replace("\n","")
            
            token = token.replace("."," ")
            
            token.replace("،"," ")
            
            values.append(token)
            
            token = ""
        
        elif begin == 1 and match == -1 and match2 == -1:
            
            token += line

    return values


# this func returns the content of the given file path.
def readFile(path):
    
    file = open(path,"r")
    
    text = file.read()
    
    return text;


# this func is used to parse the corpus file, it returns array of DID, Corpus, distinct words in all documents, and
# count of words in each document

def parseFile(path):
    
    file = open(path,'r')
    
    lineNum = 0
    
    list = []
    
    doc_info = []
    
    begin = 0;
    
    document = "";
    
    corpus = ""
    
    doc_length = []
    
    for line in file:
        
        lineNum += 1
        
        match = line.find('.DID')
        
        if match == 0:
            
            str = line.replace('.DID','')
            
            str = str.replace('\t','')
            
            str = str.replace('\n','')

            doc_info.append(str.lower())
            
        if begin == 1:
            
            has_match = line.find('.DID')
            
            if has_match>-1:
                
                begin = 0
                
                list.append(document)
                
                corpus+= document
                
                doc_length.append(len(nltk.word_tokenize(document)))
                
                document = ""
                
            else:
                
                document += line
        
        elif begin == 0:
            
            has_match = line.find('.Cat')
            
            if has_match>-1:
                
                begin = 1;
    
    list.append(document)
    
    corpus += document
    
    nltk_tokens = nltk.word_tokenize(corpus)
    
    distinct = set(nltk_tokens)
    
    doc_length.append(len(nltk.word_tokenize(document)))
    
    return list, doc_info, distinct, doc_length



# this function calculates term frequency and stores it array called TF_IDF_array
def calculate_TF():
    
    docID = 0
    
    for doc in corpus:
    
        words = word_tokenize(doc)
    
        for word in words:
    
            i, = np.where(distinct == word)
    
            TF_IDF_array[i, docID] += 1
    
        docID += 1
    
    return


# this function normalizes the TF by dividing TF of each doc to it's length.
def normalize_TF():
    
    tmp = 0
    
    for doc in DID:
    
        TF_IDF_array[:, tmp] /= doc_length[tmp]
    
        tmp += 1
    
    return



# this function calculates IDF for each word in corpus and stores it in array called IDF
def calculate_IDF():
    
    col = np.shape(TF_IDF_array)[0]
    
    for term in range(0, col):
    
        df = np.count_nonzero(TF_IDF_array[term, :])
    
        df = np.math.log(float(len(corpus)) / df)
    
        IDF[term, 0] = df
    
    return



# this function calculates TF-IDF score and writes it in TF_IDF_array
# multiplies IDF of each word with it's TF
def calculate_TF_IDF():
    
    cols = np.shape(TF_IDF_array)[1]
    
    rows = np.shape(TF_IDF_array)[0]
    
    for col in range(0,cols):
    
        for row in range(0,rows):
    
            TF_IDF_array[row, col] = TF_IDF_array[row, col] * (IDF[row])
    
    return




# this func returns norm of an array
def norm(a):
    
    return np.math.sqrt(np.dot(a, a))



# this func calculates cosine similarity of two arrays
def cosine_similarity(a, b):
    
    if norm(a)>0 and norm(b)>0:
        
        return np.dot(a,b) / (norm(a) * norm(b))
    
    else:
        
        return 0


#this func returns the corresponding IDF array of given query terms.
def query_vector(query_terms):
    
    query_terms = set(query_terms)
    
    n_terms, _ = TF_IDF_array.shape
    
    query_vector = np.zeros(n_terms)
    
    index = 0
    
    for word in distinct:
    
        if word in query_terms:
    
            query_vector[index] = IDF[index, 0]
    
        index+= 1
    
    return query_vector



# this function returns the related documents to given query terms
def query(query_terms):
    
    q = query_vector( query_terms)
    
    n_terms, _ = TF_IDF_array.shape
    
    results = []
    
    index = 0
    
    for doc in DID:
    
        doc_vec = TF_IDF_array[:,index]
    
        results.append((doc, cosine_similarity(q, doc_vec)))
    
        index+= 1
    
    return sorted(results, key = lambda t: t[1], reverse = True)



# this func reads judgments from given file
def parseJudgment(path):

    judges = []

    file = open(path,'r')

    for line in file:

        words = nltk.word_tokenize(line)

        judges.append(words)

    return judges




# this func evaluates given results based on gold data with precision@k measure
def evaluation(result, gold, k):

    tmp = 0

    for i in range(0, k):

        if result[i][0] in gold:

            tmp += 1

    return (tmp / k)




# this func returns the gold data related to given QID
def getGold(QID):

    row = np.shape(judge)[0]

    goldData = []

    for i in range(0,row):

       if QID == judge[i][0]:

           goldData.append(judge[i][1])

    return goldData



# this function evaluates set of given queries with given parameters which are used as k in precision@k measure
# and writes results to 'pathToWriteResults' file
def evalquery(queries, precisions):

    query_ID = 0

    f = open(pathToWriteResults,'a')

    for query in queries:

        res = query(word_tokenize(query))

        for precision in precisions:

            precisionAtK = evaluation(res, getGold(QID[query_ID]), precision)

            template = "precision @ %s  for QID %s  is: %s \n"

            f.write("\n")

            print (template %(precision ,QID[query_ID] , precisionAtK))

            f.write(template %(precision ,QID[query_ID] , precisionAtK))

            writeToFile(f,res[:20])

        query_ID += 1

    return



# this func writes results to given file
def writeToFile(f,res):

    f.write("[")

    for ID, precision in res:

        f.write("\n".join(["(%s , %s)" % (ID, precision)]))

    f.write("]\n")

    return



# this func combine two arrays of str
def combineTwoArrays(title, description):

    res = []

    tmp = len(title)

    for i in range(0,tmp):

        str = title[i]+" "

        str+= description[i]+" "

        res.append(str)

    return res


# read corpus file and parse it
text = readFile(path)
corpus, DID, distinct, doc_length = parseFile(path)


# create a TF-IDF array and fill it using above functions
TF_IDF_array = np.zeros([np.shape(distinct)[0], np.shape(DID)[0]])
IDF = np.zeros([np.shape(distinct)[0],1])
index = pd.DataFrame(TF_IDF_array, distinct, DID)

calculate_TF()
calculate_IDF()
normalize_TF()
calculate_TF_IDF()


# extract data from Query files
QID = parseXML(queryFilePath,'QID')
title = parseXML(queryFilePath,'title')
description = parseXML(queryFilePath,'description')

# extract data from judgment data file
judge = parseJudgment(judgmentPath)
titleAndDescription = combineTwoArrays(title,description)


# evaluate queris with their title, description, narrative using precision@5 precision@10 precision@20
precison = [5,10,20]
evalquery(title,precison)
evalquery(description,precison)
evalquery(titleAndDescription,precison)