import extraction
import preprocessing
import string_similarity
import pandas as pd

df_entity_raw, df_names_raw = extraction.extracting_data()
df_entity_clean, names_mun_clean = preprocessing.preprocessing_data(df_entity_raw, df_names_raw)
print(names_mun_clean)
test_names = ["HUILA - ALCALDÍA MUNICIPIO DE NEIVA",
              "SANTANDER - ALCALDÍA MUNICIPIO DE BUCARAMANGA"]
chain = string_similarity.contracting_chain(test_names, 3, df_entity_clean)

chain.to_csv('contracting_chain.csv')
print(chain.head())