import extraction
import preprocessing
import string_similarity

"""
The following code executes the contracting chain script and returns 
a CSV with the contracting chain for all interadministrative contracts of INVIAS
"""

#Extraction
df_entity_raw, df_names_raw = extraction.extracting_data()
# Preprocessing
df_entity_clean, names_mun_clean = preprocessing.preprocessing_data(df_entity_raw, df_names_raw)

test_names = ["HUILA - ALCALDÍA MUNICIPIO DE NEIVA",
              "SANTANDER - ALCALDÍA MUNICIPIO DE BUCARAMANGA",
              "VALLE DEL CAUCA - ALCALDÍA MUNICIPIO DE PALMIRA"]
# test_names = names_mun_clean
# Chain construction
chain = string_similarity.contracting_chain(test_names, 3, df_entity_clean)

# Printing csv
chain.to_csv('contracting_chain.csv')
