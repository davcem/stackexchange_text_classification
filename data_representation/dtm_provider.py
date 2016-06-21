import preprocessing.preprocessing_parameters as pp
from data_representation import dataset_content_document_provider as dcdp
import data_representation.dataset_spliter as ds

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier

import numpy as np
import matplotlib.pyplot as plt

"""
Word vectorizer for stemming.
Also some methods to test given vectorizer with dataset.
"""
DEFAULT_MIN_DF_KEY='min_df'
DEFAULT_MAX_DF_KEY='max_df'
DEFAULT_NORM_KEY='norm'

BEST_VECTORIZER_PARAMS_NB= {
                 
            pp.STACKEXCHANGE_TITLE_COLUMN:{DEFAULT_NORM_KEY: 'l2',
                                DEFAULT_MAX_DF_KEY: 0.2, DEFAULT_MIN_DF_KEY: 1},
            pp.STACKEXCHANGE_BODY_COLUMN:{DEFAULT_NORM_KEY: 'l2',
                                DEFAULT_MAX_DF_KEY: 0.2, DEFAULT_MIN_DF_KEY: 2},
            dcdp.retrieveValueForUsedFields([pp.STACKEXCHANGE_TITLE_COLUMN, 
                                          pp.STACKEXCHANGE_BODY_COLUMN]):
                 {DEFAULT_NORM_KEY: 'l2', 
                  DEFAULT_MAX_DF_KEY: 0.2, DEFAULT_MIN_DF_KEY: 2}
            }

BEST_VECTORIZER_PARAMS_DT= {
            pp.STACKEXCHANGE_TITLE_COLUMN:{DEFAULT_NORM_KEY: 'l2',
                                DEFAULT_MAX_DF_KEY: 0.2, DEFAULT_MIN_DF_KEY: 1},
            pp.STACKEXCHANGE_BODY_COLUMN:{DEFAULT_NORM_KEY: 'l1',
                                DEFAULT_MAX_DF_KEY: 0.5, DEFAULT_MIN_DF_KEY: 2},
            dcdp.retrieveValueForUsedFields([pp.STACKEXCHANGE_TITLE_COLUMN, 
                                          pp.STACKEXCHANGE_BODY_COLUMN]):
                 {DEFAULT_NORM_KEY: 'l2', 
                  DEFAULT_MAX_DF_KEY: 0.5, DEFAULT_MIN_DF_KEY: 5}
            }

#TODO: Fill out after param selection
BEST_VECTORIZER_PARAMS_SVC= {
            pp.STACKEXCHANGE_TITLE_COLUMN:{DEFAULT_NORM_KEY: 'l2',
                                DEFAULT_MAX_DF_KEY: 0.2, DEFAULT_MIN_DF_KEY: 2},
            pp.STACKEXCHANGE_BODY_COLUMN:{DEFAULT_NORM_KEY: 'l1',
                                DEFAULT_MAX_DF_KEY: 0.3, DEFAULT_MIN_DF_KEY: 2},
            dcdp.retrieveValueForUsedFields([pp.STACKEXCHANGE_TITLE_COLUMN, 
                                          pp.STACKEXCHANGE_BODY_COLUMN]):
                 {DEFAULT_NORM_KEY: 'l1', 
                  DEFAULT_MAX_DF_KEY: 0.4, DEFAULT_MIN_DF_KEY: 2}
            }

#best params for both
BEST_VECTORIZER_PARAMS_CLASSIFIERS = {
                           type(MultinomialNB()).__name__: 
                                    BEST_VECTORIZER_PARAMS_NB,
                           type(DecisionTreeClassifier()).__name__:
                                    BEST_VECTORIZER_PARAMS_DT,      
                           type(SVC()).__name__: None
                           }

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

def provide_vectorizer_params_for_classifier(classifier, used_fields):
    """
    #TODO: update docu
    """
    
    classifier_name = type(classifier).__name__
    
    #if OVR is entered
    if classifier_name == type(OneVsRestClassifier(SVC())).__name__:
        
        #for OnveVsRest we need the estimator
        classifier_name = type(classifier.estimator).__name__
    
    params = BEST_VECTORIZER_PARAMS_CLASSIFIERS[classifier_name]
    
    return params[used_fields]
        
def provide_countvectorizer_and_dtm(vectorizer_params, content):
    """
    #TODO: update docu
    Takes content (list[] of strings) and provides count_dtm and fitted
    vectorizer.
    
    :param content - The content to build count_dtm of
    
    :return c_vectorizer_fit - fitted CountVectorizer
    :return count_dtm - scipy.sparse.csr.csr_matrix (n_samples, n_features) with
    df.
    """

    c_vectorizer = CountVectorizer(analyzer='word',
                                   stop_words='english',
                                   min_df=vectorizer_params[DEFAULT_MIN_DF_KEY],
                                   max_df=vectorizer_params[DEFAULT_MAX_DF_KEY])
    
    c_vectorizer_fit = c_vectorizer.fit(content)
    count_dtm = c_vectorizer_fit.transform(content)
    
    return c_vectorizer_fit, count_dtm

def provide_idf_transformer_and_idf_dtm(vectorizer_params,count_dtm):
    """
    #TODO: update docu
    Takes count_dtm and transforms it to idf_dtm.
    
    :param count_dtm - scipy.sparse.csr.csr_matrix (n_samples, n_features)
    
    :return tfidf_transformer_fit - fitted TfidfTransformer
    :return dtm_idf - scipy.sparse.csr.csr_matrix (n_samples, n_features) with 
    idf.
    """
    
    tfidf_transformer = TfidfTransformer(use_idf=True, sublinear_tf=False, 
                                norm=vectorizer_params[DEFAULT_NORM_KEY])
    
    tfidf_transformer_fit = tfidf_transformer.fit(count_dtm)
    
    #the document_term_matrix with idf*tf
    dtm_idf = tfidf_transformer_fit.transform(count_dtm)
        
    return tfidf_transformer_fit, dtm_idf

def print_details_for_vectorizer(vectorizer, content, print_feature_freq=False):
    #TODO: update docu
    dtm = vectorizer.transform(content)
    vocabulary = vectorizer.vocabulary_
    feature_names = vectorizer.get_feature_names()
    print("Vectorizer transformed:")
    print("number_stop_words: " + str(len(vectorizer.get_stop_words())))
    print("number_features: " + str(len(feature_names)))
    print("dtm.shape: " + str(dtm.shape))
    
    #within count vectorizer the dtm is kept in sparse matrix, convert it to
    #np.ndarray and sum over axis to get the plain document frequencies
    plain_dfs = np.sum(dtm.toarray(),axis=0)
    print("min_df:"+str(np.min(plain_dfs))+", max_df:"+str(np.max(plain_dfs))+\
                        ", mean_df:"+str(np.mean(plain_dfs))+\
                        ", median_df:"+str(np.median(plain_dfs)))
    
    plot_histogramm_of_document_frequencies(plain_dfs)
    
    unique, counts = np.unique(plain_dfs, return_counts=True)
    print("Frequency of values [value count]")
    print(np.asarray((unique, counts)).T)

    if print_feature_freq:
        print_feature_frequencies(feature_names, vocabulary, dtm)

    return vectorizer, dtm

def print_feature_frequencies(feature_names, vocabulary, dtm):
    #TODO: update docu
    print("features and their counts:")

    for feature_name in feature_names:
        #index of feature in vocabulary
        index = vocabulary[feature_name]
        counts = dtm[:,index].todense()
        #sum over counts to get true df
        print(feature_name +":"+ str(np.sum(counts)))
        
def plot_histogramm_of_document_frequencies(plain_dfs):
    #TODO: update docu
    plt.hist(plain_dfs)
    plt.title("Document frequencies")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.show()
            
def provide_train_and_test_idf_dtms(vectorizer_params, train_content, 
                                    test_content):
    """
    #TODO: update docu
    Takes train and test data and transform it to idf document term matrices.
    
    :param train_content - Training content as List[] of strings (len=n_instances)
    :param test_content - Test content as List[] of strings (len=n_instances)
    
    :return idf_dtm_train - scipy.sparse.csr.csr_matrix.shape(n_instances train, 
    n_features)
    :return idf_dtm_test - scipy.sparse.csr.csr_matrix.shape(n_instances test, 
    n_features)
    """
        
    c_vectorizer_fit, count_dtm_train = provide_countvectorizer_and_dtm(
                                                    vectorizer_params,train_content)
    
    tfidf_transformer_fit, idf_dtm_train = provide_idf_transformer_and_idf_dtm(
                                                            vectorizer_params,
                                                            count_dtm_train)
    
    count_dtm_test = c_vectorizer_fit.transform(test_content)
    
    idf_dtm_test = tfidf_transformer_fit.transform(count_dtm_test)
    
    return idf_dtm_train, idf_dtm_test, c_vectorizer_fit

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
    
def test_provide_train_and_test_idf_dtms_real_data():
    
    """
    This method should help you to find a suitable min_df value to choose
    which features (=words) are elminated from the vectorizer
    """
    
    num_instances=100
    classifier = DecisionTreeClassifier()
    
    dataset_document_name=ds.DEFAULT_DEVOLOPMENT_DATASET_DOCUMENT_NAME
    dataset_name_train=ds.DEFAULT_TRAININGSET_NAME
    dataset_name_test=ds.DEFAULT_TESTSET_NAME
    
    for document_fields in dcdp.DEFAULT_ALL_DOCUMENT_FIELDS:
        
        document_fields = dcdp.DEFAULT_ALL_DOCUMENT_FIELDS[0]
        used_fields = dcdp.retrieveValueForUsedFields(document_fields)
        document_train = dcdp.getDatasetContentDocumentFromDatabase(
                                                dataset_document_name, 
                                                dataset_name_train, 
                                                used_fields)
        document_test = dcdp.getDatasetContentDocumentFromDatabase(
                                                dataset_document_name, 
                                                dataset_name_test, 
                                                used_fields)
            
        train_content=document_train[dcdp.DSCD_FIELD_CONTENT]
        test_content=document_test[dcdp.DSCD_FIELD_CONTENT]
        
        vectorizer_params=provide_vectorizer_params_for_classifier(classifier,
                                                                   used_fields)
    
        idf_dtm_train,idf_dtm_test=provide_train_and_test_idf_dtms(
                                                            vectorizer_params,
                                                                train_content, 
                                                                test_content)
        
        print(idf_dtm_train.shape)
        print(idf_dtm_test.shape)