from apache_couchdb import couchdb_parameters as cp
from apache_couchdb import couch_database as db

from preprocessing import preprocessor

from collections import Counter
import datetime
import re

FIELD_CREATION_DATE = 'creation_date'
FIELD_DBNAME = 'dbname'
FIELD_CONTENT = 'content'
FIELD_USED_FIELDS = 'used_fields'
FIELD_COMMENT = 'comment'
FIELD_WORD_DICTIONARY = 'wd'
FIELD_DOCUMENT_TYPE = 'document-type'

def getFieldForWordDictionaryByDocumentField(field):
    
    return FIELD_WORD_DICTIONARY + '-' + field

def addWordDictionariesToDocumentsOfDatabase(db_name, fields):
    
    """
    Creates an word dictionary (word:occurences) from all documents in given
    db_name of given fields.
    
    :param db_name: The name of database to add word dictionaries
    :param fields: The fields of documents to use for word dictionary
    """
    
    database = db.couch_database(db_name)  
    all_docs = database.getAllDocumentsFromDatabase()
    pattern = preprocessor.getWhiteListPattern()
    
    for row in all_docs.rows:
      
        document = row.doc
        
        for field in fields:
          
            counts = Counter()
            
            #sadly view _all_docs also gives design docs, so we have to check
            if field in row.doc.keys():
                
                doc_field = document[field]
                
                for word in doc_field.split():
                    counts.update(word.lower() for word in re.split(pattern,word))
    
                word_dictionary =  dict(counts)
                new_field = getFieldForWordDictionaryByDocumentField(field)
                document[new_field] = word_dictionary
               
        database.updateDocumentInDatabase(document)

def createAndInsertWordStatisticsDocumentIntoDatabase(comment, db_name, fields):
    
    """
    Creates and inserts word statistic document from given database.
    
    :param comment: The comment to insert the document with into database
    :param db_name: Name of database to generate the word dictionary of
    :param fields: The fields for the word dictionary to be used
    """ 
    
    word_dictionary = createWordDictionaryFromDatabaseContent(db_name, fields)
    
    statistics_document = buildStatisticDatabaseDocument(db_name, 
                                                         word_dictionary,fields,
                                                        comment)
    
    insertWordDictionaryIntoDatabase(cp.COUCHDB_STASTICS_NAME, 
                                     statistics_document)

def createWordDictionaryFromDatabaseContent(db_name, fields):
    
    """
    Creates an word dictionary (word:occurences) from all documents in given
    db_name of given fields.
    
    :param db_name: The name of database to create word dictionary from
    : param fields: The fields of documents to use for word dictionary
    """
    
    database = db.couch_database(db_name)
    
    all_docs = database.getAllDocumentsFromDatabase()
    
    counts = Counter()
    
    #TODO replace
    pattern = preprocessor.getWhiteListPattern()
    
    for row in all_docs.rows:
        
        document = row.doc
        
        for field in fields:
            
            #sadly view _all_docs also gives design docs, so we have to check
            if field in row.doc.keys():
                
                doc_field = document[field]
                
                for word in doc_field.split():
                    counts.update(word.lower() for word in re.split(pattern,word))
    
    word_dictionary =  dict(counts)
    
    return word_dictionary

def buildStatisticDatabaseDocument(dbname, word_dictionary, 
                                   used_fields, comment):
    
    """Builds a statistic document from given params.
    
    :param dbname - The db to use for db field in document
    :param word_dictionary - The content of the statistics document
    :param used_fields - The used fields for the creation of word dictionary
    :param comment - The comment for the statistic document (to recognize the
    content)
    """
    
    document = {}
    document[FIELD_CREATION_DATE] = str(datetime.datetime.now())
    document[FIELD_USED_FIELDS] = used_fields
    document[FIELD_DBNAME] = dbname
    document[FIELD_CONTENT] = word_dictionary
    document[FIELD_COMMENT] = comment
    document[FIELD_DOCUMENT_TYPE] = 'word_dictionary'    
    return document

def insertWordDictionaryIntoDatabase(db_name, document):
    
    """
    Insert a a worddictionary document into db with given name
    
    :param db_name: The name of database to insert document to
    :param document: The document to insert into database
    """
       
    database = db.couch_database(db_name)
        
    database.insertDocumentIntoDatabase(document)
    
def getAllStoredWordDictionaryDocumentsFromDatabase(db_name):
    
    """
    Loads the word statistic documents from given database and returns list
    of word dictionaries
    
    :param db_name: The database to load the word statistics from
    
    :return [list] of {word_dictionaries}
    """
    
    database = db.couch_database(db_name)
    all_docs = database.getAllDocumentsFromDatabase()
    word_dictionary_list = []
    
    for row in all_docs.rows:
    
        document = row.doc
        
        if row.doc[cp.COUCHDB_DOCUMENT_FIELD_ID] != cp.DESIGN_DOCUMENT_ID:
            document = row.doc           
            word_dictionary_list.append(document)
    
    return word_dictionary_list

def printDetailsOfGivenWordDictionaryDocument(wd_document):
    
    """
    Prints statistic for given word dictionary to standard out.
    
    :param word_dictionary - The word dictionary to print statistics from.
    """

    print("Word dictionary was created from database: " \
          + wd_document[FIELD_DBNAME])
    print("Word dictionary was created with comment: " \
          + wd_document[FIELD_COMMENT])
    print("Word dictionary was created with the following fields used: " \
          + str(wd_document[FIELD_USED_FIELDS]))
    print("Word dictionary was created on: " \
          + wd_document[FIELD_CREATION_DATE])