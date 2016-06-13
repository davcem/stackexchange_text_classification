import data_representation.dataset_spliter as ds
from data_representation import dtm_builder

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from sklearn.grid_search import GridSearchCV

"""
"""

import warnings
warnings.filterwarnings("ignore")

def testApproachForAllFields(min_df):
    
    dataset_document_names = [ds.DEFAULT_DATASET_DOCUMENT_NAME]
    dataset_name=ds.DEFAULT_TRAININGSET_NAME
    
    for dataset_document_name in dataset_document_names:
    
        print("Used dataset document used: " + str(dataset_document_name))
        print("Used dataset: " + str(dataset_name))
        print("---------------------------------------------------------------")
        
        for document_fields in dtm_builder.DEFAULT_ALL_DOCUMENT_FIELDS:
    
            used_fields = dtm_builder.retrieveValueForUsedFields(document_fields)
            
            print("Used fields: " + str(used_fields))
        
            document = dtm_builder.getDatasetContentDocumentFromDatabase(
                                            dataset_document_name, dataset_name, 
                                            used_fields)
            
            performGridSearchTree(document, min_df)

            print()

def performGridSearch(document, min_df):
    
    t_vectorizer = TfidfVectorizer(analyzer='word',stop_words='english', 
                                   min_df=min_df)
    
    dtm, targets = dtm_builder.buildDTMAndTargetsOfDatasetContentDocument(
                                                        document,t_vectorizer)
        
    model_to_set = OneVsRestClassifier(SVC(decision_function_shape='ovr'))

    parameters = {
    "estimator__C": [1,3,5,8],
    "estimator__kernel": ["poly","rbf", "linear", "sigmoid"],
    "estimator__degree":[1,3,5],
    "estimator__random_state":[1,10,42]
    }
    
    model_tunning = GridSearchCV(model_to_set, param_grid=parameters,n_jobs=2)

    model_tunning.fit(dtm, targets)
    
    best_parameters, score, _ = max(model_tunning.grid_scores_, key=lambda x: x[1])
    
    print(model_tunning.best_score_)
    print (model_tunning.best_params_)
    
    for param_name in sorted(parameters.keys()):
        print("%s: %r" % (param_name, best_parameters[param_name]))
        
def performGridSearchTree(document, min_df, number_instances=1000):
    
    t_vectorizer = TfidfVectorizer(analyzer='word',stop_words='english', 
                                   min_df=min_df)
    
    dtm, targets = dtm_builder.buildDTMAndTargetsOfDatasetContentDocument(
                                                        document,t_vectorizer)
    
    
    model_to_set = DecisionTreeClassifier()

    parameters = {
        "max_features": ["auto", "sqrt", "log2"],
        "random_state": [1,5,20],
    }

    model_tunning = GridSearchCV(model_to_set, param_grid=parameters, verbose=1,
                                 n_jobs=-1, scoring='f1')

    model_tunning.fit(dtm[:number_instances], targets[:number_instances])
    
    print(model_tunning.best_score_)
    print (model_tunning.best_params_)
    
    best_parameters, score, _ = max(model_tunning.grid_scores_, key=lambda x: x[1])
    
    for param_name in sorted(parameters.keys()):
        print("%s: %r" % (param_name, best_parameters[param_name]))
        
min_df=0.00003125
#min_df=0.0001       
testApproachForAllFields(min_df=min_df)