import data_representation.dataset_spliter as ds
from data_representation import dtm_provider
from data_representation import dataset_content_document_provider as dcdp
from classification import classifier_param_selection

from sklearn.naive_bayes import MultinomialNB
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn import tree
from sklearn import metrics
from sklearn import cross_validation
from sklearn.cross_validation import KFold

import warnings

warnings.filterwarnings("ignore")

"""

#What is open?
    - Perform classification with SVC - OPEN
        - Adapt params --> for title and body and title - DONE
    - Perform classification with Decission tree - OPEN
        - Adapt params --> for title and body and title - DONE
    - Cross-validation - DONE (Only test)
            http://scikit-learn.org/stable/modules/cross_validation.html
            http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.cross_val_score.html#sklearn.cross_validation.cross_val_score
    - Evaluation: http://scikit-learn.org/stable/modules/model_evaluation.html - DONE
    - Show more information to features - min_df, count_df, ausgeschlossene Features
        
    Perform classification with given classifier (see 
    #http://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html)

"""

def perform_classifiers_for_given_fieldslist(document_fields_list,
                                             baseline=True, use_tree=False):
    """
    #TODO: update docu
    Performs selected classifiers for all fields of stackexchange documents.
    
    :param baseline - whether or not to perform classifier MultinomialNB (if not
    at least SVM is performed if tree is not used)
    :param use_tree - whether or not to perform classifier DecissionTree
    :param use_cv - whether or not to also perform classifier with 
    cross validation
    """
    
    #dataset documents we want to use
    dataset_document_names = [ds.DEFAULT_DATASET_DOCUMENT_NAME]
    #training part of dataset
    dataset_name_train=ds.DEFAULT_TRAININGSET_NAME
    #testing part of dataset
    dataset_name_test=ds.DEFAULT_TESTSET_NAME
    
    #perform classification given datasets
    for dataset_document_name in dataset_document_names:
    
        print("Dataset document used: " + str(dataset_document_name))
        print("---------------------------------------------------------------")
        
        #perform classifiction for given combinations of fields
        for document_fields in document_fields_list:
                        
            #used to retrieve correct document and fields
            used_fields = dcdp.retrieveValueForUsedFields(document_fields)
            
            print("Used fields: " + str(used_fields))
            
            #get the dtm for train
            document_train = dcdp.getDatasetContentDocumentFromDatabase(
                                    dataset_document_name, dataset_name_train, 
                                    used_fields)
            #get the dtm for test
            document_test = dcdp.getDatasetContentDocumentFromDatabase(
                                    dataset_document_name, dataset_name_test, 
                                    used_fields)
            #baseline classifier?
            if baseline:
                
                estimator = MultinomialNB()
                classifier = OneVsRestClassifier(estimator)
                params = classifier_param_selection.\
                    provide_best_params_for_classifier(classifier, used_fields)
                estimator=MultinomialNB(**params)
                classifier = OneVsRestClassifier(estimator)
                
            else:#if not baseline classifier use tree or SVC
                
                if use_tree:
                    classifier = tree.DecisionTreeClassifier()
                    params = classifier_param_selection.\
                    provide_best_params_for_classifier(classifier, used_fields)

                    classifier = tree.DecisionTreeClassifier(**params)
                    
                else:
                    estimator = SVC()
                    classifier = OneVsRestClassifier(estimator)
                    params = classifier_param_selection.\
                    provide_best_params_for_classifier(classifier, used_fields)
                    estimator = SVC(decision_function_shape='ovr',**params)
                    classifier = OneVsRestClassifier(estimator)
                        
            perform_classifier(classifier,used_fields,document_train, 
                               document_test)
            
            print()
            
def perform_classifier(classifier,used_fields,document_train,document_test):
    """
    #TODO: update docu
    Perform classification with given classifier. Get the given vectorizer and
    build document term matrix with it. Then fit and perform classifier for
    train data (only for presentation purpose). Then predict test set.
    For both the train and test set metrics are printed
    
    :param classifier - The classifier to use (see
    :param document_train - dtm for training
    :param document_test - dtm for testing
    :param min_df - The minimal number of document frequency (see 
    #sklearn.feature_extraction.text import TfidfVectorizer)
    :param use_stemmer - Whether or not to use a stemmer within word tokenizer
    :param use_cv - Whether or not to use cross validation.
    """
    print("Classifier:"+str(classifier))
    
    #obtain X.data
    document_train_content = document_train[dcdp.DSCD_FIELD_CONTENT]
    document_test_content = document_test[dcdp.DSCD_FIELD_CONTENT]
    
    #num_instances=1000
    
    #document_train_content=document_train_content[:num_instances]
    #document_test_content=document_test_content[:num_instances]
    
    #first: get the best params for the given classifier and used_fields
    vectorizer_params=dtm_provider.provide_vectorizer_params_for_classifier(
                                                                classifier,
                                                                   used_fields)
    
    idf_dtm_train, idf_dtm_test,c_vectorizer_fit=dtm_provider.\
                                provide_train_and_test_idf_dtms(
                                vectorizer_params, document_train_content, 
                                document_test_content)
                                
    print(c_vectorizer_fit)
    
    #obtain targets
    targets_train = dcdp.buildTargetsFromDatasetContentDocument(document_train)
    targets_test = dcdp.buildTargetsFromDatasetContentDocument(document_test)
    
    #num_test_instances=1000
    #idf_dtm_train=idf_dtm_train[:num_instances]
    #idf_dtm_test=idf_dtm_test[:num_test_instances]
    #targets_train=targets_train[:num_instances]
    #targets_test=targets_test[:num_test_instances]
    
    feature_names = c_vectorizer_fit.get_feature_names()
    print("Number of instances(train):" + str(idf_dtm_train.shape[0]))
    print("Number of features:" + str(len(feature_names))) 
    
    classifier.fit(idf_dtm_train, targets_train)

    targets_train_predicted = classifier.predict(idf_dtm_train)
    
    print_classifier_metrics(targets_train,targets_train_predicted, "train")

    targets_test_predicted = classifier.predict(idf_dtm_test)
    
    print("Number of instances(test):" + str(idf_dtm_test.shape[0]))
    
    print_classifier_metrics(targets_test, targets_test_predicted, "test")

#TODO: Remove only needed in param_selection(GridSearch)
def perform_classifier_cross_validation(classifier, dtm_train,targets_train,
                                                 dtm_test, targets_test):
    cv = 3
    k_fold = KFold(len(targets_train), n_folds=cv,shuffle=True, 
                                        random_state=42)
    scoring = 'f1_macro'
    scores = cross_validation.cross_val_score(classifier, dtm_train, 
                                    targets_train,cv=k_fold, 
                                    scoring=scoring)
    
    print("Same classifier with cross validation:")
    print("Scores for folds" +"("+str(cv)+"):"+ str(scores))
    print(scoring + ": %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    
    targets_train_predicted = cross_validation.cross_val_predict(classifier, 
                                            dtm_train,targets_train, cv=cv)
    
    print_classifier_metrics(targets_train,targets_train_predicted, 
                               "train-with-cv")

    targets_test_predicted = cross_validation.cross_val_predict(classifier, 
                                                    dtm_test,targets_test,cv=cv)
    
    print_classifier_metrics(targets_test, targets_test_predicted, 
                               "test-with-cv")
    
    return classifier
    
def print_classifier_metrics(ytrue, ypredict,dataset_type):
    
    print("Classification performance metrics:")
    print("Metrics for dataset:" + str(dataset_type))
    print("Accuracy:" + str(metrics.accuracy_score(ytrue, ypredict, 
                                                   normalize=False)))
    print("Accuracy(Normalized):" + str(metrics.accuracy_score(ytrue, ypredict)))
    print("Precision:" + str(metrics.precision_score(ytrue, ypredict, 
                                                      average='micro')))
    print("Precision(Macro):" + str(metrics.precision_score(ytrue, ypredict, 
                                                      average='macro')))
    print("Recall:" + str(metrics.recall_score(ytrue, ypredict,
                                               average='micro')))
    print("Recall(Macro):" + str(metrics.recall_score(ytrue, ypredict, 
                                                      average='macro')))
    print("F1:" + str(metrics.f1_score(ytrue, ypredict, average='micro')))
    print("F1(Macro):" + str(metrics.f1_score(ytrue, ypredict, 
                                              average='macro')))
                                    
def perform_classification():
    
    document_fields_list=dcdp.DEFAULT_ALL_DOCUMENT_FIELDS
    #pop title field
    #document_fields_list.pop(0)
    #print(document_fields_list)
       
    #SVM baseline=True
    perform_classifiers_for_given_fieldslist(document_fields_list,
                                             baseline=False,use_tree=False)
    
    #NaiveBayes(baseline=True)
    #perform_classifiers_for_given_fieldslist(document_fields_list,baseline=True,
    #                                         use_tree=False)
    
    #DecisionTree
    #perform_classifiers_for_given_fieldslist(document_fields_list,
    #                                            baseline=False,use_tree=True)
    
perform_classification()