import json, requests
from apache_couchdb import couchdb_parameters

"""
Functionality to perform queries to apache lucene over couchdb.
"""

#URL of index
URL = 'http://localhost:5984/_fti/local/{couch_db_name}/_design/search/'
#Limit of search results
QUERY_LIMIT = '100000'
#VIEW for searching in title
TITLE_VIEW = 'by_title'
#VIEW for searching in body
BODY_VIEW = 'by_body'
#VIEW for searching in tags
TAGS_VIEW = 'by_tags'

QUERY_DETAILS_VIEW = 'view'
QUERY_DETAILS_SEARCHTERM = 'searchterm'
QUERY_DETAILS_DATABASE = 'database'

QUERY_ROWS_KEY = 'rows'

def performQueryOnLuceneCouchdbIndex(view, searchterm, 
                                     dbname=couchdb_parameters.COUCHDB_NAME, 
                                     include_docs=False):
    """
    Function performs a request for given view with given searchterm
    
    :param view: The view to query (e.g. TITLE_VIEW, BODY_VIEW, TAGS_VIEW)
    :param searchterm: The term to search in view for
    :param dbname: The couchdb to query (default: couchdb_parameters.COUCHDB_NAME)
    
    :return data: json result rows ({row['score'], row['id'])}
    """
    
    url_format = URL.format(couch_db_name=dbname)
    
    full_url = url_format + view
    
    params = dict(
                  limit=QUERY_LIMIT,
                  q=searchterm,
                  include_docs=include_docs
    )
    
    response = requests.get(url=full_url, params=params)
   
    data = json.loads(response.text)
    
    request_details = build_request_details(view, searchterm, dbname)
    
    if QUERY_ROWS_KEY in data.keys():
        
        return data['rows'], request_details
    
    else:
        
        print("Your query has no results!")
        
        return None, None

def build_request_details(view, searchterm, database):
    
    """
    Builds dictionary of request_details for the performed lucene query
    
    :param view: The view which was used for the query
    :param searchterm: Searched term
    :param database: Database to perform search in
    
    :return request_details - Dictionary{}
    """
    
    request_details = {}
    request_details[QUERY_DETAILS_VIEW] = view
    request_details[QUERY_DETAILS_SEARCHTERM] = searchterm
    request_details[QUERY_DETAILS_DATABASE] = database
    
    return request_details