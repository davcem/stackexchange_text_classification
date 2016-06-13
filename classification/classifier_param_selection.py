import preprocessing.preprocessing_parameters as pp
import data_representation.dataset_spliter as ds
from data_representation import dtm_builder
from data_representation import word_vectorizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from sklearn.grid_search import GridSearchCV

"""
"""

import warnings
warnings.filterwarnings("ignore")

DEFAULT_NUM_INSTANCES = 2000

def performGridSearchForAllFields(number_instances, use_tree=False):
    
    dataset_document_names = [ds.DEFAULT_DATASET_DOCUMENT_NAME]
    dataset_name=ds.DEFAULT_TRAININGSET_NAME
    
    for dataset_document_name in dataset_document_names:
    
        print("Used dataset document used: " + str(dataset_document_name))
        print("Used dataset: " + str(dataset_name))
        print("---------------------------------------------------------------")
        
        for document_fields in dtm_builder.DEFAULT_ALL_DOCUMENT_FIELDS:
            
            if pp.STACKEXCHANGE_TITLE_COLUMN in document_fields:
            
                min_df=word_vectorizer.DEFAULT_MIN_DF_DICT[
                                                pp.STACKEXCHANGE_TITLE_COLUMN]
            
            else:
            
                min_df=word_vectorizer.DEFAULT_MIN_DF_DICT[
                                word_vectorizer.DEFAULT_MIN_DF_DICT_KEY_OTHERS]
    
            used_fields = dtm_builder.retrieveValueForUsedFields(document_fields)
            
            print("Used fields: " + str(used_fields))
        
            document = dtm_builder.getDatasetContentDocumentFromDatabase(
                                            dataset_document_name, dataset_name, 
                                            used_fields)
            
            if use_tree:
            
                classifier = DecisionTreeClassifier()
                
                classifier_parameters = {
                                        "max_features": ["auto", "sqrt", "log2"],
                                        "random_state": [1,5,20,42]
                                        }
                
                print("DecisionTree:")

            else:
                
                classifier = OneVsRestClassifier(SVC(
                                                decision_function_shape='ovr'))
                     
                classifier_parameters = {
                  "estimator__C": [1,3,8],
                  "estimator__kernel": ["poly","rbf", "linear", "sigmoid"],
                  "estimator__degree":[1,3,5],
                  "estimator__random_state":[1,10,42]
                }
                
                print("SVM:")

                
            performGridSearchForGivenClassifier(document, classifier, 
                                                classifier_parameters,min_df,
                                                number_instances)
            
            print()
            
def performGridSearchForGivenClassifier(document, classifier, clf_parameters, 
                                min_df, number_instances):

    t_vectorizer = TfidfVectorizer(analyzer='word',stop_words='english', 
                                   min_df=min_df)
    
    dtm, targets = dtm_builder.buildDTMAndTargetsOfDatasetContentDocument(
                                                        document,t_vectorizer)
        
    param_tunning = GridSearchCV(classifier, param_grid=clf_parameters,verbose=1,
                                 n_jobs=-1, scoring='f1')
    
    if number_instances == 0:#if "0" assume not set, use all instances
        
        number_instances=dtm.shape[0]
        
    print("Use instances: " + str(number_instances))

    param_tunning.fit(dtm[:number_instances], targets[:number_instances])
    
    print("Best score: " + str(param_tunning.best_score_))
    print("Best params: " + str(param_tunning.best_params_))
    
    best_parameters, score, _ = max(param_tunning.grid_scores_, 
                                    key=lambda x: x[1])
    
    for param_name in sorted(clf_parameters.keys()):
        print("%s: %r" % (param_name, best_parameters[param_name]))

#perform grid search for all fields and for both classifiers with all instances
#number_intances=0 use all
number_instances=0
performGridSearchForAllFields(number_instances, use_tree=False)
print()
performGridSearchForAllFields(number_instances, use_tree=True)