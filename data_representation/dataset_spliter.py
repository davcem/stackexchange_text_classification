import apache_couchdb.couch_database as db
import apache_couchdb.couchdb_parameters as cp
import preprocessing.preprocessing_parameters as pp

from sklearn import cross_validation
from sklearn.preprocessing import MultiLabelBinarizer

"""
Functionality to split the dataset into training and testset and score
the corresponding document ids in document in apache couchdb
"""

DEFAULT_DB_NAME = cp.COUCHDB_CLEANED_NAME
DEFAULT_STATISTICS_DB_NAME = cp.COUCHDB_STASTICS_NAME
DEFAULT_DATASET_DOCUMENT_NAME = DEFAULT_DB_NAME + '-splitted_datasets'

#For develoment reasons build small development set to test assumptions
DEFAULT_DEVOLOPMENT_DATASET_DOCUMENT_NAME = DEFAULT_DB_NAME + '-splitted_development-datasets'

DEFAULT_TRAININGSET_NAME = 'training'
DEFAULT_TESTSET_NAME = 'test'

DEFAULT_TRAININGSET_SIZE = 0.8
DEFAULT_DEVELOPMENTSET_SIZE = 0.05

DEFAULT_DATASET_LIST_INDEX_TRAINING = 0
DEFAULT_DATASET_LIST_INDEX_TEST = 1

DATASET_DOCUMENT_FIELD_NAME = '_id'
DATASET_DOCUMENT_FIELD_TYPE = 'type'
DATASET_DOCUMENT_TYPE = 'dataset'
DATASET_DOCUMENT_FIELD_NAME_TRAINING_SET = DEFAULT_TRAININGSET_NAME
DATASET_DOCUMENT_FIELD_NAME_TEST_SET = DEFAULT_TESTSET_NAME

#TODO: Also add split of training set for development set (0.1 of training)
def perform_train_test_split(db_name=DEFAULT_DB_NAME,
                                        train_size=DEFAULT_TRAININGSET_SIZE):
    
    """
    Get all document_ids of given database and split's it according to given
    train_size.
    
    :param db_name: Name of database to split documents (default DEFAULT_DB_NAME)
    :param train_size: Size in percentage [0,1] of the training set.
    :return splitted_dataset - List of lists 
    [[DEFAULT_DATASET_LIST_INDEX_TRAINING], [DEFAULT_DATASET_LIST_INDEX_TEST]]
    """
    
    database = db.couch_database(db_name)
    all_doc_ids = database.getAllDocumentIdsFromDatabase()
   
    splitted_dataset = cross_validation.train_test_split(all_doc_ids, 
                                               train_size=train_size,
                                               random_state=42)
    return splitted_dataset

#TODO: try train_test_split again, maybe with bugfix version of sklearn
def performa_special_train_test_split(db_name=DEFAULT_DB_NAME,
                                        train_size=DEFAULT_TRAININGSET_SIZE):
    
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
            
            tags_list = document_tags.split(sep=pp.STACKEXCHANGE_TAG_SPLIT_SEPARATOR)
            
            for tag in tags_list:
                
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

def insertSplittedDatasetToDatabase(dataset_name, splitted_dataset, 
                                db_name=DEFAULT_STATISTICS_DB_NAME):
    
    """
    Stores the given dataset with given dataset_name into given database.
    
    :param dataset_name - Name of dataset to store
    :param splitted_dataset - The dataset to store as list of lists 
                        [[DEFAULT_DATASET_LIST_INDEX_TRAINING][DEFAULT_DATASET_LIST_INDEX_TEST]]
    :param db_name: Name of database to split documents.
    """
    
    database = db.couch_database(db_name)
    
    dataset_document = {}
    dataset_document[DATASET_DOCUMENT_FIELD_NAME] = dataset_name
    dataset_document[DATASET_DOCUMENT_FIELD_TYPE] = DATASET_DOCUMENT_TYPE
    dataset_document[DATASET_DOCUMENT_FIELD_NAME_TRAINING_SET] = \
                                splitted_dataset[DEFAULT_DATASET_LIST_INDEX_TRAINING]
    dataset_document[DATASET_DOCUMENT_FIELD_NAME_TEST_SET] = \
                                splitted_dataset[DEFAULT_DATASET_LIST_INDEX_TEST]
                                        
    database.insertDocumentIntoDatabase(dataset_document)
    
def getDatasetsOfDatasetDocumentFromDatabase(
                            dataset_document_name=DEFAULT_DATASET_DOCUMENT_NAME, 
                            db_name=DEFAULT_STATISTICS_DB_NAME):
    
    """
    Gets the given dataset from database.
    
    :param dataset_document_name - Name of dataset to get from database
    :param db_name: Name of database to get dataset from.
    
    :return datasets - List of lists 
                    [[DEFAULT_DATASET_LIST_INDEX_TRAINING], 
                    [DEFAULT_DATASET_LIST_INDEX_TEST]]
    """
                                
    database = db.couch_database(db_name)
    
    document = database.getDocumentFromDatabase(dataset_document_name)
    
    datasets = []
    
    datasets.insert(DEFAULT_DATASET_LIST_INDEX_TRAINING, 
                   document[DATASET_DOCUMENT_FIELD_NAME_TRAINING_SET])
    datasets.insert(DEFAULT_DATASET_LIST_INDEX_TEST, 
                   document[DATASET_DOCUMENT_FIELD_NAME_TEST_SET])
    
    return datasets

def getDatasetByNameFromDatasetDocumentByNameFromDatabase(
                        dataset_document_name=DEFAULT_DATASET_DOCUMENT_NAME,
                        dataset_name=DATASET_DOCUMENT_FIELD_NAME_TRAINING_SET):
    
    """
    Gets the dataset with given dataset_name
    
    :param dataset_name - List [] with dataset_name to retrieve dataset for (see 
    DATASET_DOCUMENT_FIELD_NAME_TRAINING_SET,DATASET_DOCUMENT_FIELD_NAME_TEST_SET)
    
    :return dataset_name - List of dataset_name corresponding to 
    dataset_name to be used.
    """
    
    datasets = getDatasetsOfDatasetDocumentFromDatabase(dataset_document_name)
    
    if DATASET_DOCUMENT_FIELD_NAME_TRAINING_SET == dataset_name:
        return datasets[DEFAULT_DATASET_LIST_INDEX_TRAINING]
    elif DATASET_DOCUMENT_FIELD_NAME_TEST_SET == dataset_name:
        return datasets[DEFAULT_DATASET_LIST_INDEX_TEST]
    
def createDevelopmentDataset():
    
    dataset = perform_train_test_split(
                                        train_size=DEFAULT_DEVELOPMENTSET_SIZE)

    insertSplittedDatasetToDatabase(DEFAULT_DEVOLOPMENT_DATASET_DOCUMENT_NAME,
                                    dataset)
    
def performDatasetSplitter():
    
    dataset_name = DEFAULT_DATASET_DOCUMENT_NAME
    
    dataset = perform_train_test_split()
    
    insertSplittedDatasetToDatabase(dataset_name, dataset)
    
    #finally also create small developmentset
    createDevelopmentDataset()