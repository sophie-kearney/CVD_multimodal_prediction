# take all genetic data files per patient

# all we need are the rsid columns (which would be the same, ti's fixed per patient)
# want to use those as columns in this genotype matrix. 
# then each patient will be a row, and the values will be the genotype calls (0,1,2)

import pandas as pd
import os
import numpy as np

# iterate through all genetic data files
genetic_data_dir = '/Users/ananyara/Github/CVD_multimodal_prediction/coherent-11-07-2022/dna'
genotype_matrix = pd.DataFrame(columns = [])


for file in os.listdir(genetic_data_dir):
    file_path = os.path.join(genetic_data_dir, file)
    df = pd.read_csv(file_path, sep=',')
    # header row for each file is: INDEX,INDEX_PREFIX,CHROMOSOME,LOCATION,STRAND,ANCESTRAL_ALLELE,VARIANT_ALLELE_LIST,GENE,CLINICAL_SIGNIFICANCE,ALLELE,VARIANT
    # we only care about: INDEX_PREFIX,CHROMOSOME,LOCATION, 
    print(f"Processing file: {file}, shape: {df.shape}")
    print(df.head())

    # there is one row per allele copy, so to get genotype dosage, we need to group by INDEX_PREFIX and sum VARIANT (which is 'true'/'false')
    # then we want to append this dosage info 
    dosage = df.groupby('INDEX_PREFIX')['VARIANT'].sum().astype(int)
    print(dosage.head())

# read in this file which has all effect sizes for SNPs:
# columns of interest: chromosome	base_pair_location	rs_id_all other_allele	effect_allele	standard_error	beta	p_value