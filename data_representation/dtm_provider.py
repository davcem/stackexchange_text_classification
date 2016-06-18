import preprocessing.preprocessing_parameters as pp
from data_representation import dataset_content_document_provider as dcdp
import data_representation.dataset_spliter as ds

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, \
TfidfTransformer
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

train_content=[        
'If you use MFC how did find it best to learn the concepts  - ', 
'Tracking hours on a project - ', 
'How do you avoid getters and setters  - ', 
'Reconciling the Boy Scout Rule and Opportunistic Refactoring with code reviews - ',
'When should I use a precondition and when to include another use-case to provide those conditions  - ',
'Best style to use multiple custom UserControls in a Grid - ', 'Do you use unit tests at work  What benefits do you get from them  - ',
'Apply filter only if not null - ',
'Best practice for parameters in asynchronous APIs in Java - ',
'Running entire frontend of a system on a flatfile cache - ',
'What conclusion to be drawn from no difference in generated assembly from 2 rather different programs  - ',
'Should foreign keys be represented directly when mapping database tables to classes  - ',
'Is there any technology similar to LINQPad for compiling C# that runs in a browser  - ',
'Tracking hours on a project - ']

test_content=[
'What s the protocol for a autoexecuting JQuery plugin  - ',
'Object behaviour or separate class  - ',
'How to select a most probable option from the list based on user text Input - ',
'Best hours use'
              ]
    
def tokenize(text):
    tokens = nltk.word_tokenize(text)
    stems = stem_tokens(tokens, stemmer)
    return stems        

def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed
        
def provide_countvectorizer_and_dtm(content):
    """
    Takes content (list[] of strings) and provides count_dtm and fitted
    vectorizer.
    
    :param content - The content to build count_dtm of
    
    :return c_vectorizer_fit - fitted CountVectorizer
    :return count_dtm - scipy.sparse.csr.csr_matrix (n_samples, n_features) with
    df.
    """
    
    c_vectorizer = CountVectorizer(analyzer='word',
                                   stop_words='english', max_df=4, min_df=2)
    
    c_vectorizer_fit = c_vectorizer.fit(content)
    count_dtm = c_vectorizer_fit.transform(content)
    
    return c_vectorizer_fit, count_dtm

def provide_idf_transformer_and_idf_dtm(count_dtm):
    """
    Takes count_dtm and transforms it to idf_dtm.
    
    :param count_dtm - scipy.sparse.csr.csr_matrix (n_samples, n_features)
    
    :return tfidf_transformer_fit - fitted TfidfTransformer
    :return dtm_idf - scipy.sparse.csr.csr_matrix (n_samples, n_features) with 
    idf.
    """
    
    tfidf_transformer = TfidfTransformer(use_idf=True, sublinear_tf=False, norm='l1')
    
    tfidf_transformer_fit = tfidf_transformer.fit(count_dtm)
    
    #the document_term_matrix with idf*tf
    dtm_idf = tfidf_transformer_fit.transform(count_dtm)
    
    print(type(dtm_idf))
        
    return tfidf_transformer_fit, dtm_idf

def print_details_for_vectorizer(vectorizer, content):

    dtm = vectorizer.transform(content)
    vocabulary = vectorizer.vocabulary_
    feature_names = vectorizer.get_feature_names()
    print("Vectorizer transformed:")
    print("number_stop_words: " + str(len(vectorizer.get_stop_words())))
    print("num_feature_names: " + str(len(feature_names)))
    print("dtm.shape: " + str(dtm.shape))
    
    print_feature_frequencies(feature_names, vocabulary, dtm)

    return vectorizer, dtm

def print_feature_frequencies(feature_names, vocabulary, dtm):
    
    print("features and their counts:")

    for feature_name in feature_names:
        #index of feature in vocabulary
        index = vocabulary[feature_name]
        counts = dtm[:,index].todense()
        #sum over counts to get true df
        print(feature_name +":"+ str(np.sum(counts)))
            
def provide_train_and_test_idf_dtms(train_content, test_content):
    """
    Takes train and test data and transform it to idf document term matrices.
    
    :param train_content - Training content as List[] of strings (len=n_instances)
    :param test_content - Test content as List[] of strings (len=n_instances)
    
    :return idf_dtm_train - scipy.sparse.csr.csr_matrix.shape(n_instances train, 
    n_features)
    :return idf_dtm_test - scipy.sparse.csr.csr_matrix.shape(n_instances test, 
    n_features)
    """
        
    c_vectorizer_fit, count_dtm_train = provide_countvectorizer_and_dtm(train_content)
    
    tfidf_transformer_fit, idf_dtm_train = provide_idf_transformer_and_idf_dtm(
                                                                 count_dtm_train)
    
    count_dtm_test = c_vectorizer_fit.transform(test_content)
    
    idf_dtm_test = tfidf_transformer_fit.transform(count_dtm_test)
    
    return idf_dtm_train, idf_dtm_test

def test_provide_train_and_test_idf_dtms():
    """
    Only small test to assure that both idf_dtms are alligned (n_features of
    both matrices should be equal)
    """
    
    idf_dtm_train, idf_dtm_test = provide_train_and_test_idf_dtms(train_content, 
                                                                  test_content)
    n_features_train=idf_dtm_train.shape[1]
    n_features_test=idf_dtm_test.shape[1]
    
    assert n_features_train == n_features_test
    
    print("shape train: " + str(idf_dtm_train.shape))
    print("shape test: " + str(idf_dtm_test.shape))
    
def temp_bug_fix():
    
    """
    This method should help you to find a suitable min_df value to choose
    which features (=words) are elminated from the vectorizer
    """

    dataset_document_name=ds.DEFAULT_DATASET_DOCUMENT_NAME
    dataset_name_train=ds.DEFAULT_TRAININGSET_NAME
    dataset_name_test=ds.DEFAULT_TESTSET_NAME
    
    document_fields = dcdp.DEFAULT_ALL_DOCUMENT_FIELDS[0]
    
    used_fields = dcdp.retrieveValueForUsedFields(document_fields)
    
    document_train = dcdp.getDatasetContentDocumentFromDatabase(
                                            dataset_document_name, dataset_name_train, 
                                            used_fields)
    
    #print(document_train['_id'])
    
    document_test = dcdp.getDatasetContentDocumentFromDatabase(
                                            dataset_document_name, dataset_name_test, 
                                            used_fields)
        
    train_content=document_train[dcdp.DSCD_FIELD_CONTENT]
    test_content=document_test[dcdp.DSCD_FIELD_CONTENT]
    train_targets=dcdp.buildTargetsFromDatasetContentDocument(document_train)
    test_targets=dcdp.buildTargetsFromDatasetContentDocument(document_test)
    
    print(document_test['_id'])
    
    #len of contents ok
    """print(len(train_content))
    print(len(test_content))
    print(len(train_targets))
    print(len(test_targets))"""
    
    idf_dtm_train, idf_dtm_test=provide_train_and_test_idf_dtms(train_content, 
                                                                test_content)
    #shape of train ok
    """print(idf_dtm_train.shape)
    print(idf_dtm_test.shape)"""
    
    #targets are aligned
    targets_train=dcdp.buildTargetsFromDatasetContentDocument(document_train)
    targets_test=dcdp.buildTargetsFromDatasetContentDocument(document_test)
    
    """print(type(targets_train))
    print(targets_train.shape)
    print(type(targets_test))
    print(targets_test.shape)"""
    
    """for (x,y), value in np.ndenumerate(targets_train):
        print(targets_train[x,y])"""
        
    
        
    """targets correct - OKAY
    document index = 20
    tags=<asynchronous-programming><message-queue><messaging><akka>
    index asynchronous-programming 48
    index message-queue 49
    index messaging 50
    index akka 51
    
    row = targets_test[20, :]
    
    for (x), value in np.ndenumerate(row):
        if value==1:
            print(x)
            print(value)
    """
    
    
    
    
def test_provide_train_and_test_idf_dtms_real_data():
    
    """
    This method should help you to find a suitable min_df value to choose
    which features (=words) are elminated from the vectorizer
    """
    
    num_instances=100
    
    for document_fields in dcdp.DEFAULT_ALL_DOCUMENT_FIELDS:
        
        """if pp.STACKEXCHANGE_TITLE_COLUMN in document_fields:
            
            min_df=DEFAULT_MIN_DF_DICT[pp.STACKEXCHANGE_TITLE_COLUMN]
            
        else:
            
            min_df=DEFAULT_MIN_DF_DICT[DEFAULT_MIN_DF_DICT_KEY_OTHERS]"""

        dataset_document_name=ds.DEFAULT_DEVOLOPMENT_DATASET_DOCUMENT_NAME
        dataset_name=ds.DEFAULT_TRAININGSET_NAME
        
        used_fields = dcdp.retrieveValueForUsedFields(document_fields)
                    
        document = dcdp.getDatasetContentDocumentFromDatabase(
                                            dataset_document_name, dataset_name, 
                                            used_fields)
        
        document_contents = document[dcdp.DSCD_FIELD_CONTENT]
        
        print(len(document_contents[:num_instances]))

    
temp_bug_fix()