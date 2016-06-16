import apache_couchdb.couch_database as db
import apache_couchdb.couchdb_parameters as cp
import preprocessing.preprocessing_parameters as pp
import data_representation.dataset_spliter as ds

import numpy as np

"""
Builds dataset content documents, stores and gets this document from database
and helps building the document term matrix and target values for given dataset.
"""

DEFAULT_DATABASE = cp.COUCHDB_CLEANED_NAME
DEFAULT_DATASET_NAME = ds.DEFAULT_TRAININGSET_NAME
DEFAULT_STATISTICS_DB_NAME = cp.COUCHDB_STASTICS_NAME
DEFAULT_DEVELOPMENT_DSCD_NAME = 'development_dataset'

#DSCD = dataset content document
DSCD_FIELD_NAME = cp.COUCHDB_DOCUMENT_FIELD_ID
DSCD_FIELD_DATASET_DOCUMENT_USED = 'dataset_document_used'
DSCD_FIELD_DATASET_NAME = 'dataset_name'
DSCD_FIELD_TYPE = 'type'
DSCD_FIELD_USED_FIELDS = 'used_fields'
DSCD_FIELD_INDEX_DICTIONARY = 'index_dictionary'
DSCD_FIELD_CONTENT = 'dataset_content'
DSCD_FIELD_CONTENT_TAGS = 'dataset_content_tags'
DSCD_FIELD_TAG_INDEX_DICTIONARY = 'tag_index_dictionary'
DSCD_FIELD_TAGS_LIST = 'dataset_tags'

DEFAULT_DSCD_NAME = 'dataset_content'
DEFAULT_DSCD_TYPE= 'dataset_content_document'

#List of lists of the different document fields
DEFAULT_ALL_DOCUMENT_FIELDS = [
                               [pp.STACKEXCHANGE_TITLE_COLUMN],
                               [pp.STACKEXCHANGE_BODY_COLUMN],
                               [pp.STACKEXCHANGE_TITLE_COLUMN,pp.STACKEXCHANGE_BODY_COLUMN]
                               ]

def buildDatasetContentListsOfDataset(dataset, document_fields):
    
    """
    Builds the dataset content list of given dataset. This is the structure we
    use later for the sklearn Vectorizer (for document term matrix)
    
    :param dataset - Dataset to build dataset content list from
    :parum document_fields - List of Lists [[]] for the fields of document 
    to use.
    
    :return index_dictionary - Dictionary{} key:docId, values:index in 
            dataset_content_list
    :return dataset_content - The content (from document fields #document_fields 
            of all documents as list
    :return tag_index_dictionary - Dictionary{] keys:tag, values: index in 
            taglist
    :return dataset_content_tags - All tags of given documents
    :return tags_list - List[] of all tags
    """
        
    database = db.couch_database(DEFAULT_DATABASE)
    all_doc_ids = database.getDocumentsForGivenIds(dataset)
    
    #stores an item for content of document
    dataset_content = []
    dataset_content_tags = []
    #key=document_id, value=index in content list
    index_dictionary = {} 
    #key=tag, value=index in tag_list
    tag_index_dictionary = {}
    #list of tags
    tags_list = []
    
    for row in all_doc_ids.rows:
        
        document = row.doc
        document_id = document.id
        document_content = ''
        
        #"sum" the text for an document over all fields
        
        #workaround to check that title and body exist
        if pp.STACKEXCHANGE_TITLE_COLUMN in document.keys():
        
            for doc_field in document_fields:
                
                if doc_field in document.keys():
                
                    document_content = document_content + document[doc_field] + ' - '
            
            dataset_content.append(document_content)
            index_dictionary[document_id]=len(dataset_content)-1
        
        if pp.STACKEXCHANGE_TAGS_COLUM in document.keys():
            
            document_tags = document[pp.STACKEXCHANGE_TAGS_COLUM]
            
            dataset_content_tags.append(document_tags)
            
            tags_split = document_tags.split(sep=pp.STACKEXCHANGE_TAG_SPLIT_SEPARATOR)
            
            for item in tags_split:
                
                #remove the closing tag (last item)
                tag = item[:-1]
                
                # tag does not exist in dict, add it to list and dictionary
                if tag not in tag_index_dictionary.keys():
                    tags_list.append(tag)
                    tag_index_dictionary[tag]=len(tags_list)-1
            
    return index_dictionary, dataset_content, tag_index_dictionary, \
        dataset_content_tags, tags_list
        
def createAndInsertDatasetContentDocuments(dataset_document_name):
    
    """
    Creates and inserts the dataset content document for the given dataset.
    Creation is performed for training and testset if they exist in the dataset
    document.
    
    :param dataset_document_name - Name of dataset document to use for creating
    the dataset content document
    
    """
    
    datasets = ds.getDatasetsOfDatasetDocumentFromDatabase(
            dataset_document_name)
    
    document_fields = [
                    [pp.STACKEXCHANGE_TITLE_COLUMN],
                    [pp.STACKEXCHANGE_BODY_COLUMN],
                    [pp.STACKEXCHANGE_TITLE_COLUMN,pp.STACKEXCHANGE_BODY_COLUMN]
                    ]
    
    #we need the index to find out if its is the training or test set for the
    #name of the dataset to store it
    for index in range(len(datasets)):
        
        dataset = datasets[index]
        
        if index == ds.DEFAULT_DATASET_LIST_INDEX_TRAINING:
            
            dataset_name=ds.DEFAULT_TRAININGSET_NAME
            
        elif index == ds.DEFAULT_DATASET_LIST_INDEX_TEST:
            
            dataset_name=ds.DEFAULT_TESTSET_NAME
    
        for fields in document_fields:
            
            index_dictionary, documents_content, tag_index_dictionary, dataset_tags,\
            tags_list = buildDatasetContentListsOfDataset(dataset, fields)
            
            used_fields = retrieveValueForUsedFields(fields)
            
            insertDatasetContentDocumentInDatabase(dataset_document_name, 
                                                   dataset_name, used_fields, 
                                                   index_dictionary,
                                                   documents_content, 
                                                   tag_index_dictionary, 
                                                   dataset_tags,tags_list)
            

def insertDatasetContentDocumentInDatabase(dataset_document_used, dataset_name, 
                                           used_fields, index_dictionary,
                                           document_content,
                                           tag_index_dictionary, document_tags,
                                           tags_list, 
                                           db_name=DEFAULT_STATISTICS_DB_NAME):
    document = {}
    document[DSCD_FIELD_DATASET_DOCUMENT_USED] = dataset_document_used
    document[DSCD_FIELD_DATASET_NAME] = dataset_name
    document[DSCD_FIELD_USED_FIELDS] = used_fields
    document[DSCD_FIELD_TYPE] = DEFAULT_DSCD_TYPE
    document[DSCD_FIELD_INDEX_DICTIONARY] = index_dictionary
    document[DSCD_FIELD_CONTENT] = document_content
    document[DSCD_FIELD_TAG_INDEX_DICTIONARY] = tag_index_dictionary
    document[DSCD_FIELD_CONTENT_TAGS] = document_tags
    document[DSCD_FIELD_TAGS_LIST] = tags_list
    
    database = db.couch_database(db_name)
    id = database.insertDocumentIntoDatabase(document)
    
    print(id)
    
def retrieveValueForUsedFields(document_fields):
    
    """
    Function is used to retrieve identical value for document_fields
    In the end only join the different fields to string...
    
    :param document_fields - List of lists [[]] of the document_fields to join.
    
    :return str - String of document_fields to use for storing and retrieving
    content document.
    """
    
    return ','.join(document_fields)

def getDatasetContentDocumentFromDatabase(dataset_document_name, dataset_name, 
                                          used_fields,doc_type=DEFAULT_DSCD_TYPE, 
                                          db_name=DEFAULT_STATISTICS_DB_NAME):  
    """
    
    Gets the required dataset content list from database
    
    :param dataset_document_name - The name of dataset_document the dataset
    content document was created of.
    :param dataset_name - The dataset to retrieve (training|test)
    :param used_fields - The used fields for creating the dataset content doc
    :parm doc_type - Default set to DEFAULT_DSCD_TYPE
    :param db_name - Default set to DEFAULT_STATISTICS_DB_NAME
    
    :return dataset_content_document
    
    """

    database = db.couch_database(db_name)
    #we need all documents from the statistics db
    all_docs = database.getAllDocumentsFromDatabase()
    
    #we only need the one with used fields
    
    for row in all_docs.rows:
        
        document = row.doc
        
        document_keys = document.keys()
        
        #check if all needed fields exist
        if DSCD_FIELD_DATASET_DOCUMENT_USED in document_keys \
            and DSCD_FIELD_TYPE in document_keys \
            and DSCD_FIELD_DATASET_NAME in document_keys \
            and DSCD_FIELD_USED_FIELDS in document_keys:
            
            #the document we want needs to have the matching 
            #dataset_document_name type, dataset_name and the correct 
            #used fields            
            if  dataset_document_name == document[DSCD_FIELD_DATASET_DOCUMENT_USED]\
                and doc_type == document[DSCD_FIELD_TYPE] \
                and dataset_name == document[DSCD_FIELD_DATASET_NAME] \
                and used_fields == document[DSCD_FIELD_USED_FIELDS]:
                
                return document

#TODO: Remove            
def buildDTMAndTargetsOfDatasetContentDocumentOld(document, vectorizer):
    
    """
    Builds the document-term matrix and targets of a dataset content document.
    
    :param document - The dataset content document to build dtm and targets 
    from
    :param vectorizer - The vectorizer to use for build dtm and targets
    
    :return dtm - The document term matrix
    :return targets - The target tags of document term matrix
    """     
    document_contents = document[DSCD_FIELD_CONTENT]
    
    targets = buildTargetsFromDatasetContentDocument(document)
    
    Y = vectorizer.fit(document_contents)
    
    dtm = Y.transform(document_contents)
    
    return dtm, targets
    
def buildTargetsFromDatasetContentDocument(document):
    
    """
    Builds the document-term matrix and targets of a dataset content document.
    
    :param document - The dataset content document to build targets from.
    
    :return targets - The targets for given document.
    
    """
    
    index_dictionary = document[DSCD_FIELD_INDEX_DICTIONARY]
    tag_index_dictionary = document[DSCD_FIELD_TAG_INDEX_DICTIONARY]
    document_content_tags = document[DSCD_FIELD_CONTENT_TAGS]
    document_tags = document[DSCD_FIELD_TAGS_LIST]
    
    #TODO: Fix this - This workaround is needed because we can not assure
    #that the classes (tags) in our training and test set are equaly splitted:
    #E.g. We have 1000 tags in train but only 990 in test --> to assure 
    #functionality of classifiere targets of train and test need to have at
    #least the same shape. To achive that, for test set get corresponding 
    #training set and create targets with number of columns corresponding to
    #number of tags within the training data
    
    """if document[DSCD_FIELD_DATASET_NAME] == ds.DEFAULT_TESTSET_NAME:
                
        obtain_train = getDatasetContentDocumentFromDatabase(
                document[DSCD_FIELD_DATASET_DOCUMENT_USED],
                ds.DEFAULT_TRAININGSET_NAME, 
                document[DSCD_FIELD_USED_FIELDS])
        
        document_tags = obtain_train[DSCD_FIELD_TAGS_LIST]"""
    
    targets = np.zeros((len(index_dictionary.keys()),len(document_tags)))
    
    for key in index_dictionary.keys():
        
        index = index_dictionary[key]
        document_tags = document_content_tags[index]
        tags_split = document_tags.split(sep=pp.STACKEXCHANGE_TAG_SPLIT_SEPARATOR)
        
        for item in tags_split:
                
            #remove the closing tag (last item)
            tag = item[:-1]
            tag_index = tag_index_dictionary[tag]
            targets[index,tag_index] = 1
            
    return targets
    
def performDTMBuilder():
    
    dataset_document_names = [ds.DEFAULT_DEVOLOPMENT_DATASET_DOCUMENT_NAME, 
                         ds.DEFAULT_DATASET_DOCUMENT_NAME]
    
    for dataset_document_name in dataset_document_names:
        
        createAndInsertDatasetContentDocuments(dataset_document_name)