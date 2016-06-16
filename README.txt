* Packages:
	** apache_couchdb - Sources dealing with apache couch_db
	** preprocessing - Sources dealing with importing and initial cleaning of data
	** feature engineering - Sources dealing with feature selection and feature building
	
* HowTo:
** You'll need to have some raw dataset from stackexchange stored as .csv
*** Store it in the resources/ folder
** Adapt the parameters in apache_couchdb.couchdb_parameters.py:
*** COUCHDB_URL - URL of your couchdb installation
*** COUCHDB_NAME - Name of "default" couchdb to use
*** Other Couchdbs for different stages of process
** Adapt the parameters in preprocessing.preprocessing_parameters.py
*** STACKEXCHANGE_RAW_FILE - Name of the raw exported .csv file
*** STACKEXCHANGE_JSON_FILE - Name of the .json file
*** STACKEXCHANGE_*_COLUMN - The column names of your data
** Run preprocessor.fillCouchDBWithRawData() - fill your couch database with data: 

* Useful hints
* One advantage of couchdb is the possibility to replicate databases
** This functionality is used within 
	apache_couchdb.couch_database.replicateCouchDBToNewCouchDB():
*** This has the advantage that everytime you perform some operations (cleaning)
on the database you can replicate it, to potentialy restore it
* Out of the box apache-couchdb has no full index search --> to "implement"
this functionality the couchdb-lucene application was installed
** For the application to full index the database it must be started and
and special view (desgin document) for the apache-couchdb has to be inserted
*** This functionality is implemented in apache_couchdb.couch_database.insertDesignDocumentFromGivenPath()
** To perfrom search on the index of the apache-couchdb see 
apache-couchdb.couchdb_lucene_query.performQueryOnLuceneCouchdbIndex()

* Preprocessing - cleaning
** At the moment the words are split only by whitespace ''
* With 10000 Top words
Number different words: 300743
Number word occurrences: 7251645
Number different words: 10000
Number word occurrences: 6566129
Percentage of top words represent: 3.3250981735235734
Occurrences of top words represent: 90.54675180596955
** Next try with words splittet

*Databases:
- stackexchange-raw: raw-database
- stackexchange-clean_html: raw-database - html tags removed
- stackexchange-clean: stackexchange-clean_html and fields cleaned