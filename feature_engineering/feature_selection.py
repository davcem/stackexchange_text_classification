import apache_couchdb.couch_database as db
import apache_couchdb.couchdb_parameters as cp

import preprocessing.preprocessing_parameters as pp
import feature_engineering.word_dictionary as wd

def performFeatureSelection():
    
    #1.) replicate our current clean database to new database
    database = db.couch_database(cp.COUCHDB_CLEANED_NAME)
    
    #2.) Now we work with new clean database
    db_name = cp.COUCHDB_CLEANED_WD_NAME
    database.replicateCouchDBToNewCouchDB(db_name)
    
    #3.) Define the fields we want to create word dictionaries of
    fields = [pp.STACKEXCHANGE_TITLE_COLUMN, pp.STACKEXCHANGE_BODY_COLUMN]
    
    #4) Add the word dictionaries of given fields to documents
    wd.addWordDictionariesToDocumentsOfDatabase(db_name, fields)
    
performFeatureSelection()