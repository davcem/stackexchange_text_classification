import preprocessing.preprocessing_parameters as pp
import data_representation.dataset_spliter as ds
from data_representation import dataset_content_document_provider as dcdp

from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from time import time

import warnings

warnings.filterwarnings("ignore")

"""
This class should help you to find the best parameters for your classifiers.
Within the constants #BEST_PARAMS_SVM, #BEST_PARAMS_DECISIONTREE and 
#BEST_VECTORIZER_PARAMS_CLASSIFIERS there are already the best params for the original
used Stackexchange dataset.
To test different parameters simply start #perform_gridsearch_for_given_fieldslist
and adapt param ranges.
"""

#TODO: adapt after grid search
BEST_PARAMS_NB = {pp.STACKEXCHANGE_TITLE_COLUMN:{'alpha': 0.001},
              pp.STACKEXCHANGE_BODY_COLUMN:{'alpha': 0.01},
              dcdp.retrieveValueForUsedFields([pp.STACKEXCHANGE_TITLE_COLUMN, 
                                          pp.STACKEXCHANGE_BODY_COLUMN]):{
                                            'alpha': 0.01}
              }

#best params for decisiontree
BEST_PARAMS_DECISIONTREE = {pp.STACKEXCHANGE_TITLE_COLUMN:{'random_state': 42, 
                                     'splitter': 'best', 
                                     'criterion': 'gini', 
                                     'min_samples_split': 1, 
                                     'max_features': 'auto'},
                        pp.STACKEXCHANGE_BODY_COLUMN:{'random_state': 42, 
                                'splitter': 'random', 
                                'criterion': 'gini', 
                                'min_samples_split': 1, 
                                'max_features': 'auto'},
                        dcdp.retrieveValueForUsedFields(
                                        [pp.STACKEXCHANGE_TITLE_COLUMN, 
                                          pp.STACKEXCHANGE_BODY_COLUMN]):{
                                      'random_state': 1, 
                                      'splitter': 'best', 
                                      'criterion': 'gini', 
                                      'min_samples_split': 1, 
                                      'max_features': 'auto'}
                        }

#best params for svm
BEST_PARAMS_SVM = {pp.STACKEXCHANGE_TITLE_COLUMN:{'gamma': 0.3, 
                                    'random_state': 42,'kernel': 'linear', 
                                    'C': 3.0,'max_iter': 5000, 'tol': 0.01},
              pp.STACKEXCHANGE_BODY_COLUMN:{'gamma': 0.3, 
                                    'random_state': 42,'kernel': 'linear', 
                                    'C': 3.0,'max_iter': 1000, 'tol': 0.01},
              dcdp.retrieveValueForUsedFields([pp.STACKEXCHANGE_TITLE_COLUMN, 
                                          pp.STACKEXCHANGE_BODY_COLUMN]):{
                            'gamma': 0.3,'random_state': 1,'kernel': 'linear', 
                            'C': 3.0,'max_iter': 1000, 'tol': 1e-06}
              }

#best params for both
BEST_VECTORIZER_PARAMS_CLASSIFIERS = {
                           type(MultinomialNB()).__name__:  BEST_PARAMS_NB,
                           type(DecisionTreeClassifier()).__name__:
                                    BEST_PARAMS_DECISIONTREE,      
                           type(SVC()).__name__: BEST_PARAMS_SVM
                           }
   
def perform_gridsearch_for_given_fieldslist(document_fields_list,
                                            number_instances,svm=False):
    #TODO: update docu
    
    dataset_document_names = [ds.DEFAULT_DEVOLOPMENT_DATASET_DOCUMENT_NAME]
    dataset_name=ds.DEFAULT_TRAININGSET_NAME
    
    for dataset_document_name in dataset_document_names:
    
        print("Used dataset document used: " + str(dataset_document_name))
        print("Used dataset: " + str(dataset_name))
        print("===============================================================")
        
        for document_fields in document_fields_list:

            used_fields = dcdp.retrieveValueForUsedFields(document_fields)
            
            train_doc = dcdp.getDatasetContentDocumentFromDatabase(
                                            dataset_document_name, dataset_name, 
                                            used_fields)
            
            
            train_data = train_doc[dcdp.DSCD_FIELD_CONTENT]
            train_targets = dcdp.buildTargetsFromDatasetContentDocument(train_doc)
            
            if number_instances==0:#if number of instances is not set,take 
                #whole set
                number_instances=len(train_data)
                
            print("Train for num_instances:"+str(number_instances))
            
            print("Used_field(s):"+str(used_fields))
            print("-----------------------------------------------------------")
            
            if svm:           
                # svm
                print("SVM")
                print("-----")
                classifier = OneVsRestClassifier(SVC(decision_function_shape='ovr'))
                         
                classifier_parameters = {
                      "clf__estimator__C": [1.0, 3.0],
                      "clf__estimator__kernel": ["poly", "rbf", "linear", "sigmoid"],
                      "clf__estimator__gamma":[0.3, 0.01],
                      "clf__estimator__tol":[1e-3, 1e-6],
                      "clf__estimator__random_state":[42],
                      "clf__estimator__max_iter":[1000,5000]
                }
                    
                perform_gridsearch_for_classifier(classifier,
                                                classifier_parameters, train_data,
                                                train_targets, number_instances)
                print()
            
            # naive bayes
            print("Naive Bayes")
            print("-----------")
            estimator = MultinomialNB()
            classifier = OneVsRestClassifier(estimator)
        
            classifier_parameters = {
                                        'clf__estimator__alpha': 
                                                        (1e-2, 1e-3, 1e-4, 1e-5, 1e-6,
                                                         1e-7)
                                        }
                
            perform_gridsearch_for_classifier(classifier,
                                                classifier_parameters, train_data,
                                                train_targets, number_instances)
            print()
                
            # decision tree
            print("Decision tree")
            print("--------------")
            classifier = DecisionTreeClassifier()
                    
            classifier_parameters = {
                                        "clf__criterion":["gini", "entropy"],
                                        "clf__splitter":["best", "random"],
                                        "clf__min_samples_split": [1, 2, 5],
                                        "clf__max_features": ["auto", "sqrt", "log2"],
                                        "clf__random_state": [1, 42]
                                        }
                                
            perform_gridsearch_for_classifier(classifier,
                                                classifier_parameters, train_data,
                                                train_targets, number_instances)
            print()
            print("***********************************************************")

def perform_gridsearch_for_classifier(classifier, classifier_parameters, 
                            train_data,train_targets,number_instances):
    
    #TODO: update docu
        
    pipeline = Pipeline([
                            ('vect', CountVectorizer(analyzer='word',
                                   stop_words='english')),
                            ('tfidf', TfidfTransformer(use_idf=True)),
                            ('clf', classifier),
                            ])
    
    pipeline_parameters = {
                    'vect__min_df': (0.001,0.0001),
                    'vect__max_df': (0.25,0.5),              
                    'tfidf__norm': ('l1', 'l2')
    }

    #"concat" the parameters of the pipeline with parameters of classifier
    overall_parameters=dict(pipeline_parameters,**classifier_parameters)
    
    #perform grid search for 2 cores and perform cross_validation with 3 folds
    grid_search = GridSearchCV(pipeline,param_grid=overall_parameters,verbose=1,
                                 n_jobs=2, scoring='f1_macro',cv=3)
    t0 = time()    
    grid_search.fit(train_data[:number_instances], train_targets[:number_instances])
    print("Grid search took %0.3fs" % (time() - t0))
    
    print("Best score: " + str(grid_search.best_score_))
    print("Best params: " + str(grid_search.best_params_))
    
    best_parameters, score, _ = max(grid_search.grid_scores_, 
                                    key=lambda x: x[1])
    
    for param_name in sorted(overall_parameters.keys()):
        print("%s: %r" % (param_name, best_parameters[param_name]))
                    
def provide_best_params_for_classifier(classifier, used_fields):
    
    """
    Function returns the stored best params for given classifier and given
    used_fields.
    
    :param classifier - The classifier object to obtain the params for.
    :param used_fields - Used fields as obtained from 
    #data_representation.dataset_spliter.retrieveValueForUsedFields
    """
    
    classifier_name = type(classifier).__name__
    
    #if OVR is entered
    if classifier_name == type(OneVsRestClassifier(SVC())).__name__:
        
        #for OnveVsRest we need the estimator
        classifier_name = type(classifier.estimator).__name__
    
    params = BEST_VECTORIZER_PARAMS_CLASSIFIERS[classifier_name]
    
    return params[used_fields]

def perform_parameter_selection():
    #TODO: perform selection only for SVM for body and title+body
    number_instances=100
    document_fields_list=dcdp.DEFAULT_ALL_DOCUMENT_FIELDS
    #pop title field
    #document_fields_list.pop(0)
    #print(document_fields_list)
    #perform grid search for all fields and for both classifiers with all instances
    #number_intances=0 use all
    perform_gridsearch_for_given_fieldslist(document_fields_list,number_instances,
                                     svm=True)
    
#perform_parameter_selection()