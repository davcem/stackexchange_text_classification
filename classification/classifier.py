import preprocessing.preprocessing_parameters as pp
import data_representation.dataset_spliter as ds
from data_representation import word_vectorizer
from data_representation import dtm_builder

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn import tree
from sklearn import metrics

import numpy as np
import warnings

warnings.filterwarnings("ignore")

"""

#TODO: What is open?
    - Perform classification with SVC - OPEN
    - Perform classification with Decission tree - OPEN
    - Evaluation: http://scikit-learn.org/stable/modules/model_evaluation.html
    - Maybe: 
        - Check if dataset should be splitted again (Low train, but high test score)
        - GridSearch for min_df
        - Cross-validation?
            http://scikit-learn.org/stable/modules/cross_validation.html
            http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.cross_val_score.html#sklearn.cross_validation.cross_val_score
        - Perform classification with RandomForest

"""

def performClassificationWithGivenClassifier(classifier,document_train, 
                                             document_test,min_df,
                                             use_stemmer=False):
    
    """
    Perform classification with given classifier (see 
    #http://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html)
    
    :param classifier - The classifier to use (see
    :param use_stemmer - Wether or not to use a stemmer within word tokenizer
    :param document_train - dtm for training
    :param document_test - dtm for testing
    :param min_df - The minimal number of document frequency (see 
    #sklearn.feature_extraction.text import TfidfVectorizer)
    """
    
    if use_stemmer:
        
        t_vectorizer = TfidfVectorizer(analyzer='word',stop_words='english', 
                            min_df=min_df, tokenizer=word_vectorizer.tokenize)
        
    else:
        t_vectorizer = TfidfVectorizer(analyzer='word',stop_words='english', 
                                   min_df=min_df)
        
    dtm_train, targets_train = \
            dtm_builder.buildDTMAndTargetsOfDatasetContentDocument(
                                                document_train,t_vectorizer)
    
    dtm_test, targets_test = \
        dtm_builder.buildDTMAndTargetsOfDatasetContentDocument(
                                                document_test,t_vectorizer)
    
    feature_names = t_vectorizer.get_feature_names()
    print("Number of features:" + str(len(feature_names))) 
    
    #put the estimator in the OVR-Classifier and fit the model
    classifier.fit(dtm_train, targets_train)
    
    score_train = classifier.score(dtm_train, targets_train)
    
    print("Score train:" + str(score_train))
    
    targets_predicted = classifier.predict(dtm_test)
    
    print("Score test:" + str(np.mean(targets_predicted == targets_test)))
    
    printClassificationMetrics(targets_test, targets_predicted)
    
def printClassificationMetrics(ytrue, ypredict):
    
    print("Classification performance metrics:")
    print("Accuracy:" + str(metrics.accuracy_score(ytrue, ypredict)))
    print("Precision:" + str(metrics.precision_score(ytrue, ypredict, 
                                                      average='micro')))
    print("Precision(Macro):" + str(metrics.precision_score(ytrue, ypredict, 
                                                      average='macro')))
    print("Recall:" + str(metrics.recall_score(ytrue, ypredict)))
    print("F1:" + str(metrics.f1_score(ytrue, ypredict, average='micro')))
    print("F1(Macro):" + str(metrics.f1_score(ytrue, ypredict, average='macro')))
            
def performClassificationForAllFields(baseline=True, use_tree=False):
    
    """
    Tests an approach for all combinations of fields:
        - only title
        - only body
        - title + body
    """
    
    #dataset documents we want to use
    dataset_document_names = [ds.DEFAULT_DATASET_DOCUMENT_NAME]
    #training part of dataset
    dataset_name_train=ds.DEFAULT_TRAININGSET_NAME
    #testing part of dataset
    dataset_name_test=ds.DEFAULT_TRAININGSET_NAME
    
    #perform classification given datasets
    for dataset_document_name in dataset_document_names:
    
        print("Used dataset document used: " + str(dataset_document_name))
        print("---------------------------------------------------------------")
        
        #perform classifiction for given combinations of fields
        for document_fields in dtm_builder.DEFAULT_ALL_DOCUMENT_FIELDS:
            
            if pp.STACKEXCHANGE_TITLE_COLUMN in document_fields:
            
                min_df=word_vectorizer.DEFAULT_MIN_DF_DICT[
                                                pp.STACKEXCHANGE_TITLE_COLUMN]
            
            else:
            
                min_df=word_vectorizer.DEFAULT_MIN_DF_DICT[
                                word_vectorizer.DEFAULT_MIN_DF_DICT_KEY_OTHERS]
            
            #used to retrieve correct document and fields
            used_fields = dtm_builder.retrieveValueForUsedFields(document_fields)
            
            print("Used fields: " + str(used_fields))
            
            #get the dtm for train
            document_train = dtm_builder.getDatasetContentDocumentFromDatabase(
                                    dataset_document_name, dataset_name_train, 
                                    used_fields)
            #get the dtm for test
            document_test = dtm_builder.getDatasetContentDocumentFromDatabase(
                                    dataset_document_name, dataset_name_test, 
                                    used_fields)
            #baseline classifier?
            if baseline:
                
                estimator = MultinomialNB(alpha=0.01)
                classifier = OneVsRestClassifier(estimator)
                
            else:#if not baseline classifier use tree or SVC
                
                if use_tree:
                    
                    classifier = tree.DecisionTreeClassifier(random_state=42)
                    
                else:    
                    estimator = estimator = SVC(C=3.0, kernel='linear', 
                                                decision_function_shape='ovr',
                                                degree=1, random_state=1)
                    
                    classifier = OneVsRestClassifier(estimator)
                
            performClassificationWithGivenClassifier(classifier, 
                                                document_train, document_test,
                                                min_df, use_stemmer=False)
            
            print()
                        
#Devlopment set - best title min_df=0.00005
"""
min_df=0.0000625
for title: 0.178613358718 
for body: 0.307187053784
body+title: 0.314548627638

min_df=0.00003125
for title: 0.272346501666 
for body: 0.300777407584 
body+title: 0.300936062193

min_df=0.000016 (kein Fortschritt mehr)
for title: 0.272346501666
for body: 0.300777407584
body+title: 0.300936062193
"""

performClassificationForAllFields(baseline=False)