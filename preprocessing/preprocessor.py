from preprocessing import csv_to_json_converter as converter
from preprocessing import preprocessing_parameters as pp

from apache_couchdb import couchdb_parameters as cp
import apache_couchdb.couch_database as db

import re

CLEANING_WHITELIST = ['\+', '\-', '#']

def getWhiteListPattern():
    
    """
    Returns pattern for cleaning.
    
    :return whitelist_pattern - RegEx pattern already compiled.
    """

    whitelist = r'[^\w ' + ''.join(CLEANING_WHITELIST) + ']|_'
       
    whitelist_pattern = re.compile(whitelist)
    
    return whitelist_pattern

def fillCouchDBWithRawData():
    
    """
    1.) Converts the raw-dataset from csv to json
    2.) Imports the json file into the raw database
    """
    
    #first convert the raw-dataset from csv to json
    converter.readRawCSVFileAndConvertItIntoJsonFile(pp.STACKEXCHANGE_RAW_FILE, 
                                                     pp.STACKEXCHANGE_JSON_FILE)
    
    #second load the json data into the couchdb
    import_file = pp.RESOURCES_FOLDER + pp.STACKEXCHANGE_JSON_FILE
    
    db_name = cp.COUCHDB_RAW_NAME
    
    readJsonFromFileAndImportIntoDatabase(db_name, import_file)
    
def readJsonFromFileAndImportIntoDatabase(db_name, file):
    
    """
    Reads json from given file and imports it into given db_name
    
    :param db_name: Name of database to import json content of file to
    :param file: File incl. name and path to import json content from
    """
    
    database = db.couch_database(db_name)
    
    database.bulkInsertDocumentsFromFile(file)

def importDesignDocumentIntoGivenDatabase(dbname):
    
    """
    Imports the default design document into given database
    
    :param db_name: Name of database to import design document to
    """
    
    #path to design_document_file
    design_document_file = pp.RESOURCES_FOLDER + pp.STACKEXCHANGE_DESIGN_DOCUMENT

    database = db.couch_database(dbname=dbname)
    
    stored_id = database.insertDesignDocumentFromGivenPath(design_document_file)
    
    print("ID of stored doc: " + str(stored_id))
        
def cleanHTMLTagsFromDocumentsInDatabase(db_name, fields):
    
    """
    Find and replace all (html-) tags from given fields in given database
    
    :param db_name - The db in which tags should be replaced
    :param fields - List fields from which html tags should be replaced
    
    """
    
    #see source: 
    #http://kevin.deldycke.com/2008/07/python-ultimate-regular-expression-to-catch-html-tags/
    ultimate_regexp = """(?i)<\/?\w+((\s+\w+(\s*=\s*(?:\".*?\"|'.*?'|[^'\">\s]+))?)+\s*|\s*)\/?>"""
    
    html_tag_pattern = re.compile(ultimate_regexp)
    
    database = db.couch_database(db_name)
    
    all_docs = database.getAllDocumentsFromDatabase()

    for row in all_docs.rows:
        
        #print("ID of document: " + str(row.id))
        
        document = row.doc
        
        for field in fields:
            
            #sadly view _all_docs also gives design docs, so we have to check
            if field in row.doc.keys():
        
                doc_field = document[field]
                
                if re.findall(html_tag_pattern, doc_field):
                    
                    cleaned_doc_field = re.sub(html_tag_pattern,'', doc_field)
                    document[field] = cleaned_doc_field
                    
        database.updateDocumentInDatabase(document)
                

def cleanFieldsOfDocumentsFromDatabase(db_name, fields):
    
    """
    Find and replace all (html-) tags from given fields in given database
    
    :param db_name - The db in which tags should be replaced
    :param fields - List fields from which html tags should be replaced
    
    """
    
    pattern = getWhiteListPattern()
    
    database = db.couch_database(db_name)
    
    all_docs = database.getAllDocumentsFromDatabase()

    for row in all_docs.rows:
        
        document = row.doc
        
        for field in fields:
            
            #sadly view _all_docs also gives design docs, so we have to check
            if field in row.doc.keys():
        
                doc_field = document[field]
                cleaned_doc_field = re.sub(pattern, ' ',doc_field)
                document[field] = cleaned_doc_field
            
        database.updateDocumentInDatabase(document)
      
def performPreprocessor():
    
    """
    Equivalent to the main method, but it needed to be called separate
    Performs all relevant preprocessing steps
    """
    import_file = pp.RESOURCES_FOLDER + pp.STACKEXCHANGE_JSON_FILE
    readJsonFromFileAndImportIntoDatabase(cp.COUCHDB_RAW_NAME,import_file)
    importDesignDocumentIntoGivenDatabase(cp.COUCHDB_RAW_NAME)
    
    db_name = cp.COUCHDB_CLEAN_HTML_NAME
    FIELDS = [pp.STACKEXCHANGE_TITLE_COLUMN, pp.STACKEXCHANGE_BODY_COLUMN]
    
    database = db.couch_database(cp.COUCHDB_RAW_NAME)
    database.replicateCouchDBToNewCouchDB(db_name)
    
    cleanHTMLTagsFromDocumentsInDatabase(db_name, FIELDS)
    
    database = db.couch_database(db_name)
    database.replicateCouchDBToNewCouchDB(cp.COUCHDB_CLEANED_NAME)
    
    db_name = cp.COUCHDB_CLEANED_NAME
    cleanFieldsOfDocumentsFromDatabase(db_name, FIELDS)