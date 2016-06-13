import couchdb
import json
import apache_couchdb.couchdb_parameters as cp

"""
Main functionality to deal with apache couchdb
"""

class couch_database(object):
    
    def __init__(self, dbname, url=cp.COUCHDB_URL):

        self.server = couchdb.Server(url)
        self.dbname = dbname
        
        try:
            
            self.db = self.server[dbname]
        
        except couchdb.http.ResourceNotFound:
            #db with name not exists --> create it
            self.db = self.server.create(dbname)
            print('database created')
            
    def createCouchDB(self):
        self.db = self.server.create(self.dbname)
    
    def bulkInsertDocumentsFromFile(self, input_file):
        
        with open(input_file) as jsonfile:
            for row in jsonfile:
                db_entry = json.loads(row)
                self.db.save(db_entry)
                
    def getAllDocumentsFromDatabase(self):
        
        all_docs = self.db.view('_all_docs', include_docs=True)   
        return all_docs
    
    def getAllDocumentIdsFromDatabase(self):
        
        all_ids = []
        
        for row in self.db.view('_all_docs'):
            
            all_ids.append(row.id)
        
        return all_ids
    
    def getDocumentsForGivenIds(self, ids_list):
        
        """
        Get documents from database by given list of ids
        
        :param ids_list - The List[] of ids to retrieve.
        
        :return all_docs - All docs for the given ids (docs included)
        """    
        
        all_docs = self.db.view('_all_docs', keys=ids_list, include_docs=True)
                
        return all_docs
    
    def insertDocumentIntoDatabase(self, document):
        stored_id = self.server[self.dbname].save(document)
        return stored_id
          
    def getDocumentFromDatabase(self,stored_id):        
        database = self.server[self.dbname]
        document = database[stored_id]
        return document
    
    def deleteAllDocumentsFromDatabase(self):
        
        docs =  self.db.view('_all_docs')
        ddocs = []
        for i in docs:
            ddocs.append({'_id':i['id'],'_rev':i['value']['rev']})
        
        self.db.purge(ddocs)
        
    def updateDocumentInDatabase(self, doc):
        
        self.db.save(doc)              
    
    def getDatabaseInfo(self, ddoc):
        
        return self.db.info(ddoc=ddoc)
    
    def copyDesignDocumentFromDBToDB(self, db_from, db_to):
        
        """
        Copies a design document from given db to other db.
        
        :param db_from - Name of database to copy design document from (source)
        :param db_to - Name of database to copy design document to (target)
        
        :return new_id - Id of copied document.
        """
        
        source = couch_database(cp.COUCHDB_URL, db_from)
        document = source.getDocumentFromDatabase(cp.DESIGN_DOCUMENT_ID)
        
        if '_rev' in document:
            del document['_rev']
        
        target = couch_database(cp.COUCHDB_URL, db_to)
        new_id = target.insertDocumentIntoDatabase(document)
        
        return new_id
    
    def insertDesignDocumentFromGivenPath(self, path_to_file):
        
        """
        Inserts design_document from path_to_file into current database
        
        :param path_to_file: The complete path and file name of the design
        document
        """
        
        with open(path_to_file) as json_file:
            json_doc = json.load(json_file)
    
        #database = db.couch_database(dbname=dbname)
        
        stored_id = self.insertDocumentIntoDatabase(json_doc)
            
        return stored_id
    
    def replicateCouchDBToNewCouchDB(self, name_target_db):
        
        """
        Replicates the actual database to the source database.
        
        :param name_target_db: The name of the newly create target to replicate the actual 
        db to
        
        :return new_id - Id of copied document.
        """
        
        self.server.replicate(self.dbname, name_target_db, continuous=False, 
                              create_target=True)