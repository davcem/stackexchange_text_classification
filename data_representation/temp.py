from data_representation import dataset_content_document_provider as dcdp

tags="<security><development-methodologies><hardware>"

tag_list=dcdp.provide_tag_list_of_tag_content(tags)

for tag in tag_list:
    
    print(tag)