#default resource folder location
RESOURCES_FOLDER = '../resources/'
#default input files
STACKEXCHANGE_RAW_FILE = 'KDDM2-Stackexchange-Dataset.csv'
STACKEXCHANGE_JSON_FILE = 'KDDM2-Stackexchange-Dataset.json'
STACKEXCHANGE_DESIGN_DOCUMENT = 'stackexchange_design_document.json'

#description of your .csv file structure (columns)
STACKEXCHANGE_TITLE_COLUMN = 'title'
STACKEXCHANGE_BODY_COLUMN = 'body'
STACKEXCHANGE_TAGS_COLUM = 'tags'
STACKEXCHANGE_DATA_COLUMNS = ['id','CreationDate',STACKEXCHANGE_TITLE_COLUMN,
                            STACKEXCHANGE_BODY_COLUMN,STACKEXCHANGE_TAGS_COLUM]

#TODO: Remove dont needed
STACKEXCHANGE_TAG_SPLIT_SEPARATOR = '>'