import apache_couchdb.couch_database as db
import apache_couchdb.couchdb_parameters as cp

import preprocessing.preprocessing_parameters as pp

from sklearn import cross_validation
from sklearn.cross_validation import LabelShuffleSplit
from sklearn.cross_validation import StratifiedShuffleSplit

from sklearn.preprocessing import MultiLabelBinarizer

from data_representation import dtm_builder
import data_representation.dataset_spliter as ds
from sklearn.feature_extraction.text import TfidfVectorizer

import operator

def testClassifier():
    
    min_df=0.00003125
    
    dataset_document_name = ds.DEFAULT_DATASET_DOCUMENT_NAME
    
    #training part of dataset
    dataset_name_train=ds.DEFAULT_TRAININGSET_NAME
    #testing part of dataset
    dataset_name_test=ds.DEFAULT_TESTSET_NAME
    
    for document_fields in dtm_builder.DEFAULT_ALL_DOCUMENT_FIELDS:
    
        #used to retrieve correct document and fields
        used_fields = dtm_builder.retrieveValueForUsedFields(document_fields)
                
        #get the dtm for train
        document_train = dtm_builder.getDatasetContentDocumentFromDatabase(
                                        dataset_document_name, dataset_name_train, 
                                        used_fields)
                
        document_test = dtm_builder.getDatasetContentDocumentFromDatabase(
                                        dataset_document_name, dataset_name_test, 
                                        used_fields)
    
        t_vectorizer = TfidfVectorizer(analyzer='word',stop_words='english', 
                                   min_df=min_df)
    
        document_train_content = document_train[dtm_builder.DSCD_FIELD_CONTENT]
        
        X_train_tfidf = t_vectorizer.fit_transform(document_train_content)
        targets_train = dtm_builder.buildTargetsFromDatasetContentDocument(document_train)
        
        """printDetails(X_train_tfidf)
        printDetails(targets_train)
        print()"""
        
        document_test_content = document_train[dtm_builder.DSCD_FIELD_CONTENT]
        
        X_test_tfidf = t_vectorizer.transform(document_test_content)
        targets_test = dtm_builder.buildTargetsFromDatasetContentDocument(document_test)
        
        """printDetails(X_test_tfidf)
        printDetails(targets_test)
        print()
        print()"""
    
        
        """dtm_train, targets_train = \
                dtm_builder.buildDTMAndTargetsOfDatasetContentDocument(
                                                    document_train,t_vectorizer)"""
        
        #t_vectorizer = TfidfVectorizer(analyzer='word',min_df=0)
        
        """dtm_test, targets_test = \
            dtm_builder.buildDTMAndTargetsOfDatasetContentDocument(
                                                    document_test,t_vectorizer)"""

        
        
            
def perform_train_test_split(db_name=ds.DEFAULT_DB_NAME,
                                        train_size=ds.DEFAULT_TRAININGSET_SIZE):
    
    """
    Get all document_ids of given database and split's it according to given
    train_size.
    The tricky part is that we n
    
    :param db_name: Name of database to split documents (default DEFAULT_DB_NAME)
    :param train_size: Size in percentage [0,1] of the training set.
    :return splitted_dataset - List of lists 
                    [[DEFAULT_DATASET_LIST_INDEX_TRAINING], 
                    [DEFAULT_DATASET_LIST_INDEX_TEST]]
    """
    
    database = db.couch_database(db_name)
    all_docs = database.getAllDocumentsFromDatabase()
    
    doc_ids_list = []
    all_tag_list = []
    
    i = 0
    
    for row in all_docs.rows:
        
        document = row.doc
        #append the document id to doc_ids_list
        doc_ids_list.append(document[cp.COUCHDB_DOCUMENT_FIELD_ID])
        
        tag_list = []
        
        #if document has tags than split and add them
        if pp.STACKEXCHANGE_TAGS_COLUM in document.keys():
            
            document_tags = document[pp.STACKEXCHANGE_TAGS_COLUM]
            
            tags_split = document_tags.split(sep=dtm_builder.TAG_SPLIT_separator)
            
            for tag in tags_split:
                
                #remove the closing tag (last item)
                tag_list.append(tag[:-1])
        #append the list of document tags to all_tag_list        
        all_tag_list.append(tag_list)
        
        i += 1
        
        if i > 10000:
            break
    
    mlb = MultiLabelBinarizer()
    tags_encoded = mlb.fit_transform(all_tag_list)

    
    print(len(doc_ids_list))
    
    splitted_dataset = cross_validation.train_test_split(doc_ids_list,tags_encoded,
                                               train_size=0.8, random_state=42, 
                                               stratify=tags_encoded)
        
def printDetails(dtm):
    
    print(dtm.shape)
    
def printDetailsDocument(document):
    
    print(document['_id'])
        
#perform_train_test_split()

testClassifier()