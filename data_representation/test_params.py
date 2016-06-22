from data_representation import dtm_provider
from data_representation import dataset_content_document_provider as dcdp
import data_representation.dataset_spliter as ds

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier

classifier=DecisionTreeClassifier()
document_fields=dcdp.DEFAULT_ALL_DOCUMENT_FIELDS[2]
used_fields = dcdp.retrieveValueForUsedFields(document_fields)
params=dtm_provider.provide_vectorizer_params_for_classifier(classifier, 
                                                             used_fields)

print(params)