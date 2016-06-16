import preprocessing.preprocessing_parameters as pp
import data_representation.dataset_spliter as ds
from data_representation import dataset_content_document_provider as dcdp
from data_representation import word_vectorizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.grid_search import GridSearchCV

"""
This class should help you to find the best parameters for your classifiers.
Within the constants #BEST_PARAMS_SVM, #BEST_PARAMS_DECISIONTREE and 
#BEST_PARAMS_CLASSIFIERS there are already the best params for the original
used Stackexchange dataset.
To test different parameters simply start #perform_gridsearch_for_all_fields
and adapt param ranges.
"""

#best params for svm
BEST_PARAMS_SVM = {'title':{'gamma': 0.3, 'random_state': 1,'kernel': 'sigmoid', 
                            'C': 8.0,'max_iter': 10000, 'tol': 0.0001},
              'body':{'gamma': 0.3, 'random_state': 1, 'kernel': 'sigmoid', 
                      'C': 8.0,'max_iter': 10000, 'tol': 0.01},
              'title,body':{'gamma': 0.3, 'random_state': 1,'kernel': 'linear', 
                            'C': 3.0,'max_iter': 10000,'tol': 0.01}
              }

#best params for decisiontree
BEST_PARAMS_DECISIONTREE = {'title':{'min_samples_split': 10, 'splitter': 'best', 
                                 'max_features': 'auto', 'random_state': 5, 
                                 'criterion': 'gini'},
                        'body':{'min_samples_split': 2, 'splitter': 'best', 
                                'max_features': 'auto', 'random_state': 1, 
                                'criterion': 'gini'},
                        'title,body':{'min_samples_split': 2, 
                                      'splitter': 'random', 
                                      'max_features': 'auto', 'random_state': 1, 
                                      'criterion': 'gini'}
                        }
#best params for both
BEST_PARAMS_CLASSIFIERS = {
                           type(DecisionTreeClassifier()).__name__:
                                    BEST_PARAMS_DECISIONTREE,
                           type(SVC()).__name__: BEST_PARAMS_SVM
                           }

DEFAULT_NUM_INSTANCES = 5000

def perform_gridsearch_for_all_fields(number_instances, use_tree=False):
    
    dataset_document_names = [ds.DEFAULT_DATASET_DOCUMENT_NAME]
    dataset_name=ds.DEFAULT_TRAININGSET_NAME
    
    for dataset_document_name in dataset_document_names:
    
        print("Used dataset document used: " + str(dataset_document_name))
        print("Used dataset: " + str(dataset_name))
        print("---------------------------------------------------------------")
        
        for document_fields in dcdp.DEFAULT_ALL_DOCUMENT_FIELDS:
            
            if pp.STACKEXCHANGE_TITLE_COLUMN in document_fields:
            
                min_df=word_vectorizer.DEFAULT_MIN_DF_DICT[
                                                pp.STACKEXCHANGE_TITLE_COLUMN]
            
            else:
            
                min_df=word_vectorizer.DEFAULT_MIN_DF_DICT[
                                word_vectorizer.DEFAULT_MIN_DF_DICT_KEY_OTHERS]
    
            used_fields = dcdp.retrieveValueForUsedFields(document_fields)
            
            print("Used fields: " + str(used_fields))
        
            document = dcdp.getDatasetContentDocumentFromDatabase(
                                            dataset_document_name, dataset_name, 
                                            used_fields)
            
            if use_tree:
            
                classifier = DecisionTreeClassifier()
                
                classifier_parameters = {
                                         "criterion":["gini", "entropy"],
                                         "splitter":["best", "random"],
                                         "min_samples_split": [2,5,10],
                                        "max_features": ["auto","sqrt", "log2"],
                                        "random_state": [1,5,20,42]
                                        }
                
                print("DecisionTree:")

            else:
                
                classifier = OneVsRestClassifier(SVC(
                                                decision_function_shape='ovr'))
                     
                classifier_parameters = {
                  "estimator__C": [1.0,3.0,8.0],
                  "estimator__kernel": ["poly","rbf", "linear", "sigmoid"],
                  "estimator__gamma":[0.3, 0.01, 0.00003],
                  "estimator__tol":[1e-2, 1e-4],
                  "estimator__random_state":[1,42],
                  "estimator__max_iter":[10000, -1]
                }
                
                print("SVM:")

                
            perform_gridsearch_for_classifier(document, classifier, 
                                                classifier_parameters,min_df,
                                                number_instances)
            
            print()
            
def perform_gridsearch_for_classifier(document, classifier, clf_parameters, 
                                min_df, number_instances):

    t_vectorizer = TfidfVectorizer(analyzer='word',stop_words='english', 
                                   min_df=min_df)
    
    document_train_content = document[dcdp.DSCD_FIELD_CONTENT]
    train_tfidf = t_vectorizer.fit_transform(document_train_content)
    targets_train = dcdp.buildTargetsFromDatasetContentDocument(document)
        
    param_tunning = GridSearchCV(classifier, param_grid=clf_parameters,verbose=1,
                                 n_jobs=2, scoring='f1_macro')
    
    if number_instances == 0:#if "0" assume not set, use all instances
        
        number_instances=train_tfidf.shape[0]
        
    print("Use instances: " + str(number_instances))

    param_tunning.fit(train_tfidf[:number_instances], targets_train[:number_instances])
    
    print("Best score: " + str(param_tunning.best_score_))
    print("Best params: " + str(param_tunning.best_params_))
    
    best_parameters, score, _ = max(param_tunning.grid_scores_, 
                                    key=lambda x: x[1])
    
    for param_name in sorted(clf_parameters.keys()):
        print("%s: %r" % (param_name, best_parameters[param_name]))
        
def get_params_for_classifier_used_fields(classifier, used_fields):
    
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
    
    params = BEST_PARAMS_CLASSIFIERS[classifier_name]
    
    return params[used_fields]

def perform_parameter_selection():

    #perform grid search for all fields and for both classifiers with all instances
    #number_intances=0 use all
    number_instances=DEFAULT_NUM_INSTANCES
    perform_gridsearch_for_all_fields(number_instances, use_tree=False)