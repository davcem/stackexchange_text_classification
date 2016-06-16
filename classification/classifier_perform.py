import preprocessing.preprocessing_parameters as pp
import data_representation.dataset_spliter as ds
from data_representation import word_vectorizer
from data_representation import dataset_content_document_provider as dcdp
from classification import classifier_param_selection

from sklearn.feature_extraction.text import TfidfVectorizer
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

DEFAULT_CROSS_VALIDATION_FOLD = 3

def perform_classifiers_for_all_fields(baseline=True, use_tree=False, 
                                      use_cv=False):
    """
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
        for document_fields in dcdp.DEFAULT_ALL_DOCUMENT_FIELDS:
            
            if pp.STACKEXCHANGE_TITLE_COLUMN in document_fields:
            
                min_df=word_vectorizer.DEFAULT_MIN_DF_DICT[
                                                pp.STACKEXCHANGE_TITLE_COLUMN]
            
            else:
            
                min_df=word_vectorizer.DEFAULT_MIN_DF_DICT[
                                word_vectorizer.DEFAULT_MIN_DF_DICT_KEY_OTHERS]
            
            #used to retrieve correct document and fields
            used_fields = dcdp.retrieveValueForUsedFields(document_fields)
            
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
                
                estimator = MultinomialNB(alpha=0.01)
                classifier = OneVsRestClassifier(estimator)
                
            else:#if not baseline classifier use tree or SVC
                
                if use_tree:
                    
                    print("Classifier:DecisionTree")
                    classifier = tree.DecisionTreeClassifier()
                    params = classifier_param_selection.\
                    get_params_for_classifier_used_fields(classifier, 
                                                          used_fields)

                    classifier = tree.DecisionTreeClassifier(**params)
                    
                else:
                    
                    print("Classifier:SVM")
                    estimator = SVC()
                    params = classifier_param_selection.\
                    get_params_for_classifier_used_fields(estimator, 
                                                          used_fields)

                    estimator = SVC(**params)
                    classifier = OneVsRestClassifier(estimator)
            
            print("Used fields: " + str(used_fields))
                
            perform_classifier(classifier, document_train, document_test,min_df, 
                               use_cv, use_stemmer=False)
            
            print()
            
def perform_classifier(classifier,document_train,document_test,min_df,
                                            use_cv=False,use_stemmer=False):
    
    """
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
    
    if use_stemmer:
        
        t_vectorizer = TfidfVectorizer(analyzer='word',stop_words='english', 
                            min_df=min_df, tokenizer=word_vectorizer.tokenize)
        
    else:
        t_vectorizer = TfidfVectorizer(analyzer='word',stop_words='english', 
                                   min_df=min_df)
        
    """dtm_train, targets_train = \
            dtm_builder.buildDTMAndTargetsOfDatasetContentDocument(
                                                document_train,t_vectorizer)
    
    dtm_test, targets_test = \
        dtm_builder.buildDTMAndTargetsOfDatasetContentDocument(
                                                document_test,t_vectorizer)"""
        
    document_train_content = document_train[dcdp.DSCD_FIELD_CONTENT]
    train_tfidf = t_vectorizer.fit_transform(document_train_content)
    targets_train = dcdp.buildTargetsFromDatasetContentDocument(
                                                                document_train)
        
    document_test_content = document_test[dcdp.DSCD_FIELD_CONTENT]
    test_tfidf = t_vectorizer.transform(document_test_content)
    targets_test = dcdp.buildTargetsFromDatasetContentDocument(
                                                                document_test)
    
    feature_names = t_vectorizer.get_feature_names()
    
    print("Number of instances(train):" + str(train_tfidf.shape[0]))
    print("Number of features:" + str(len(feature_names))) 
    
    classifier.fit(train_tfidf, targets_train)

    targets_train_predicted = classifier.predict(train_tfidf)
    
    print_classifier_metrics(targets_train,targets_train_predicted, "train")

    targets_test_predicted = classifier.predict(test_tfidf)
    
    print("Number of instances(test):" + str(test_tfidf.shape[0]))
    
    print_classifier_metrics(targets_test, targets_test_predicted, "test")
    
    if use_cv:
        perform_classifier_cross_validation(classifier, train_tfidf,targets_train,
                                                 test_tfidf, targets_test)

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

def perform_classification():
    
    #NaiveBayes
    #perform_classifiers_for_all_fields(baseline=True, use_cv=True)
    #print()
    
    #SVM
    #perform_classifiers_for_all_fields(baseline=False,use_tree=False)
    #print()
    
    #DecisionTree
    perform_classifiers_for_all_fields(baseline=False,use_tree=True,use_cv=True)
    print()
    perform_classifiers_for_all_fields(baseline=False,use_tree=False, use_cv=True)
    
perform_classification()