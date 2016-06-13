import preprocessing.preprocessing_parameters as pp
from data_representation import dtm_builder
import data_representation.dataset_spliter as ds

from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import nltk
from nltk.stem.porter import PorterStemmer

"""
Word vectorizer for stemming.
Also some methods to test given vectorizer with dataset.
"""

DEFAULT_MIN_DF_DICT_KEY_OTHERS = 'OTHERS'

DEFAULT_MIN_DF_DICT = {pp.STACKEXCHANGE_TITLE_COLUMN: 0.00003125,
                       DEFAULT_MIN_DF_DICT_KEY_OTHERS: 0.00003125}

stemmer = PorterStemmer()
    
def tokenize(text):
    tokens = nltk.word_tokenize(text)
    stems = stem_tokens(tokens, stemmer)
    return stems        

def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed

def testVectorizer():
    
    """
    This method should help you to find a suitable min_df value to choose
    which features (=words) are elminated from the vectorizer
    """
    
    for document_fields in dtm_builder.DEFAULT_ALL_DOCUMENT_FIELDS:
        
        if pp.STACKEXCHANGE_TITLE_COLUMN in document_fields:
            
            min_df=DEFAULT_MIN_DF_DICT[pp.STACKEXCHANGE_TITLE_COLUMN]
            
        else:
            
            min_df=DEFAULT_MIN_DF_DICT[DEFAULT_MIN_DF_DICT_KEY_OTHERS]

        dataset_document_name=ds.DEFAULT_DATASET_DOCUMENT_NAME
        dataset_name=ds.DEFAULT_TRAININGSET_NAME
        
        used_fields = dtm_builder.retrieveValueForUsedFields(document_fields)
            
        print("Used fields: " + str(used_fields))
        print("================================")
        print("min_df: " + str(min_df))
        
        document = dtm_builder.getDatasetContentDocumentFromDatabase(
                                            dataset_document_name, dataset_name, 
                                            used_fields)
        
        document_contents = document[dtm_builder.DSCD_FIELD_CONTENT]
        
        #normal vectorizer
        t_vectorizer = TfidfVectorizer(analyzer='word',stop_words='english', 
                                       min_df=min_df)
        
        print("Normal vectorizer:")
        print("------------------")
        
        fittransformVectorizerAndPrintDetails(t_vectorizer, document_contents)
        
        print()
        
        #vectorizer with stemmer
        t_vectorizer = TfidfVectorizer(analyzer='word',stop_words='english', 
                                       min_df=min_df, tokenizer=tokenize)
        
        print("Stemmer vectorizer:")
        print("-------------------")
        
        fittransformVectorizerAndPrintDetails(t_vectorizer, document_contents)
        
def fittransformVectorizerAndPrintDetails(t_vectorizer,document_contents):
    
    Y = t_vectorizer.fit(document_contents) 
    feature_names = t_vectorizer.get_feature_names()
    idf = t_vectorizer.idf_
    dtm_t = Y.transform(document_contents)
        
    #IDF=log(N/DF) --> high if relevant
    print("Idf: " + "min: " + str(np.min(idf)) + "; max: " + str(np.max(idf)) + 
              "; mean: " + str(np.mean(idf)) + "; median: " + str(np.median(idf)))
        
    print("Number of features: " + str(len(feature_names)))        
    print("Number of stop words: " + str(len(t_vectorizer.get_stop_words())))
    print("Shape idf: " + str(dtm_t.shape)) 