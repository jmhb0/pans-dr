"""
Functions for reading, restructuring, and displaying the raw data csv's.

This code is specific to datasets shared with this project. Some were created 
manually, and so changing there format may require adjusting some of the 
constants (filnames, row numbers) in this file. 

File paths are constructed assuming the caller is in the repo home directory. 
"""
import os 
import pandas as pd
import numpy as np
import re
import treelib 
import warnings

## Constants for data file structure. *.csv files should be in /data/ 
# Phosphophlow
fname_phosphoflow = 'all-pans-phosphoflow-data-gated-jun-29.csv'
# Proteomics
fname_proteomics = 'pans-somascan-data-jul-23.csv'
proteomics_skiprows = 54
proteomics_num_row_headers = 25
proteomics_num_col_headers = 16
# Metabolomics
fname_metabolomics = 'pans-Metabolomics.csv'
metabolomics_skiprows = 0
metabolomics_num_row_headers = 12
metabolomics_num_col_headers = 17
# Data pairing for proteomics & metabolomics 
fname_omics_pairing = 'proteomics-metabolomics-sample-pairing.csv'
omics_pairing_col_width = 20

# directory of the calling code
dir_path = os.path.dirname(os.path.realpath(__file__))
data_dir = '{}/data'.format(dir_path)

################################################################################
# Phosphoflow code
'''
Read in data. Set `id` to index. Remove duplicates. Adjust some inconsistencies
in column names ti help with processing later on. 

@param drop_rows: list/array of row indices to remove. Is for removing data
outliers. The `id` columns is the 
'''
def phosphoflow_get_data(drop_rows=None):
	indx_col = 'id'
	df = pd.read_csv('{}/{}'.format(
				data_dir
				, fname_phosphoflow
				)
			, index_col='id')
	if drop_rows is not None:
		df = d.drop(drop_rows, axis=0) 
		# df = d.reset_index()


	# Drop some cols. 
	# 1st is an exact duplicate. 
	# 2nd is approx duplicate (probably determined using a different method)
	# All the others are accounted for in the sense that 
	drop_cols = [ 'CD8/Effector CD8 | Freq. of Parent (%)' 
	            , 'PBMC | Freq. of live singles (%)'      
	            , 'mDC | Freq. of live singles (%)'       
	            , 'pDC | Freq. of live singles (%)'       
	            , 'total monocytes | Freq. of live singles (%)' 
	            ,'total monocytes/CCR2+CX3CR1+ | Freq. of Parent (%)'
	            ,'total monocytes/CCR2+CX3CR1- | Freq. of Parent (%)'
	            ,'total monocytes/CCR2-CX3CR1+ | Freq. of Parent (%)'
	            ,'total monocytes/CCR2-CX3CR1- | Freq. of Parent (%)'
	            ]     

	df = df.drop(labels=drop_cols, axis=1)

	# make some adjustments for consistency 
	df.columns = [st\
	             .replace('  ',' ')\
	             .replace('CD3+','CD3')\
	             .replace(' (%)', '')
	             for st in df.columns] # some columns had double-spacing. 

	# Subpopulations of monocyte types list 'Parent' as their cell parent, 
	# rather than their real parent. This code ajdjust this
	cols = list(df.columns)
	mon_types = ['non-classical monocytes','classical monocytes'
				, 'intermediate monocytes', 'total monocytes']
	for i in range(len(cols)):
	    for mon_type in mon_types:
	        if mon_type in cols[i]: 
	            cols[i] = cols[i].replace('Parent', mon_type)
	df.columns = cols
	return df

''' 
Get list of column names matching a regex, and pulling out the relevant marker
 + cell information. 
@param cols: list of strings to match (column names)
@param r: regular expression with 3 capturing groups. Group 1 should
  be the whole string. 
Return array of strings matching regex `r`. Return two more arrays
containing the two capture groups in r. These arrays have equal lengths
'''
def phosphoflow_get_col_indices_and_labels(r, cols):
    re_cols = np.asarray([r.match(s).groups() 
             for s in cols 
             if r.match(s) is not None])
    re_cols_indices = np.asarray([i 
                                  for i, s in enumerate(cols)
                                if r.match(s) is not None])
    names, group1, group2 = re_cols[:,0], re_cols[:,1], re_cols[:,2]
    return names, group1, group2, re_cols_indices

'''
@param hide_output: set to True to supress print statements
'''
def phosphoflow_get_data_col_meta(df, hide_output=False): 
	cols = df.columns

	# Get regular 'freq' columns
	re_freq = re.compile("^((.*)\ \|\ Freq\.\ of\ (.*))$")
	cols_population , cols_population_label1, cols_population_label2 \
		, cols_freq_indx = phosphoflow_get_col_indices_and_labels(re_freq, cols)

	# Get 'median'/expression cols
	re_median = re.compile("^((.*)\ \|\ Median\ (.*))$")
	cols_expression , cols_expression_label1, cols_expression_label2 \
		, cols_expression_indx = phosphoflow_get_col_indices_and_labels(re_median, cols)

	if not hide_output:
		print("Pulled freq data set labels with {} cols"\
			.format(len(cols_population)))
		print("Pulled med data set labels with {} cols"\
			.format(len(cols_expression)))

	
	return cols_population , cols_population_label1, cols_population_label2 \
			, cols_expression , cols_expression_label1, cols_expression_label2

'''
Get the id index from `df` and create `id_lookup` that has info 
'''
def phosphoflow_make_id_lookup(df):
	id_lookup = pd.DataFrame(df.index).set_index('id', drop=False)
	id_lookup = id_lookup['id']\
	                .str.split('-', expand=True)\
	                .rename(columns={0:"num",1:"patient",2:"state"})
	return id_lookup

# func for displaying cell_tree data
def phosphoflow_build_and_print_cell_tree(df, cols, cols_label1, cols_label2):
    cell_indices = list(np.arange(len(cols)))
    tree_size_check = np.zeros(len(cols))

    cell_tree = treelib.Tree()
    cell_tree_means = treelib.Tree() # second tree but I will print the means in the name as well 
    root = "total live singlets"
    cell_tree.create_node(root, -1)
    cell_tree_means.create_node(root, -1)

    lookup={}
    lookup[root] = -1
    freq_means = np.mean(df[cols], axis=0)

    # do a queue 
    print("number of cols = {}".format(len(cols)))
    for i in cell_indices:
        col_name = cols[i]
        cell_name = cols_label1[i]
        parent_name = cols_label2[i]

        # If parent is not already in the tree, move it to the back of the queue to be checked later.
        if parent_name not in lookup.keys():
            # First check if we've been through one cycle of the queue without adding any more nodes
            # We know by storing the size of the tree from the last time we looked at this node. 
            if tree_size_check[i] == cell_tree.size():
                print("Terminating.")
                print("Could not fit the following columns into the tree:")
                assigned_cells = np.array(list(lookup.values()))
                all_cell_indice = np.arange(len(cols))
                unassigned_cells = np.isin(all_cell_indice, assigned_cells, invert=True)
                print(cols[unassigned_cells])
                break 
            # otherwise move items to back of the queue
            else:
                tree_size_check[i] = cell_tree.size()
                cell_indices.append(i) 
                continue

        # We know this cell's parent is in the tree. Add it to the tree, and to the lookup. 
        lookup[cell_name] = i 
        cell_tree.create_node(cell_name
                         , i 
                         , parent=lookup[parent_name]
                         , data=freq_means[col_name]
                             )
        cell_tree_means.create_node("{:15s}\t\t{:.2f}".format(cell_name, freq_means[col_name])
                         , i 
                         , parent=lookup[parent_name]
                         , data=freq_means[col_name]
                            )
    print("Number of cells that are children of 'total live singlets' = {}".format(cell_tree_means.size()-1))    
    print("\n\nEntire tree with cell types + their means across all populations:\n")
    cell_tree_means.show()
    return cell_tree, lookup

################################################################################
# Proteomics code
'''
@param drop_rows: list/array of row indices to remove. Is for removing data
outliers. The `id` columns is the 
'''
def proteomics_get_data(drop_rows=None):
	with warnings.catch_warnings():
	    warnings.simplefilter(action='ignore', category=FutureWarning)
	    d = pd.read_csv('{}/{}'.format(
						data_dir
						, fname_proteomics
						)
                    , skiprows=proteomics_skiprows
                    , index_col=list(range(proteomics_num_row_headers))
                    , header=list(range(proteomics_num_col_headers))
                    )

	# Save col headers to `indx_cols`. Later will create a lookup frame this data 
	# But for the main data frame, we'll drop them, except `SeqId`
	indx_cols = d.columns
	d.columns = d.columns.droplevel(
	    list(range(1,len(d.columns.levels)))
	)

	# Save rows headers to `indx_rows`. Later will create a lookup frame this data 
	# But for the main data frame, we'll drop them, except `SampleId`
	indx_rows = d.index
	delete_indx_row = list(indx_rows.names)
	delete_indx_row.remove('SampleId')
	d = d.droplevel(delete_indx_row)\
	        .drop('SeqId',axis=1)
	d.columns.name = 'SeqId'

	d = d.drop(drop_rows, axis=0)
	d_indx_split = d.index.str.split('_')
	d_indx_splt_filt = [np.flip(np.array(z))[0] for z in d_indx_split] # gross
	d.index = d_indx_splt_filt

	return d, indx_rows, indx_cols

'''
TODO
'''
def proteomics_get_SampleId_lookup(indx_rows): 
	SampleId_lookup = indx_rows\
                    .to_frame(index=False)\
                    .set_index('SampleId')\
                    .transpose()\
                    .fillna('')
	# TODO - less hacky version of below (the code is repeated for the main df index)
	SampleId_lookup = SampleId_lookup.transpose() 
	tmp_indx = SampleId_lookup.index.str.split('_')
	tmp_indx = [np.flip(np.array(z))[0] for z in tmp_indx] # gross
	SampleId_lookup.index = tmp_indx

	return SampleId_lookup

'''
TODO
'''
def proteomics_get_SeqId_lookup(indx_cols): 
	# headers_indx was the column multiindex read from csv
	SeqId_lookup = indx_cols.to_frame(index=False)
	SeqId_lookup.columns = SeqId_lookup.iloc[0]
	SeqId_lookup = SeqId_lookup\
	                .drop(0)\
	                .set_index('SeqId')
	return SeqId_lookup 

################################################################################
# metabolomics
'''
@param drop_rows: list/array of row indices to remove. Is for removing data
outliers. The `id` columns is the 

Is slightly different from the proteomics function in that it returns `None` for
indx_rows. This is because indx_rows must re-import the csv to get the correct 
bounds.
'''
def metabolomics_get_data(drop_rows=None):
	# must do 2 reads to get all the header info
	d = pd.read_csv('{}/{}'.format(
				data_dir
				, fname_metabolomics
				)
            , skiprows=metabolomics_skiprows
            , index_col=list(range(metabolomics_num_row_headers))
            , header=list(range(metabolomics_num_col_headers))
            )

	# save index_cols to use later
	indx_cols = d.columns

	drop_cols = list(range(len(d.columns.levels)))
	keep_col_indx = 0 # sample name
	drop_cols.remove(keep_col_indx)
	d.columns = d.columns.droplevel(
	    drop_cols
	)

	d.index = d.index.droplevel(
	    list(range(1,len(d.index.levels)))
	)

	d = d.drop(['CLIENT IDENTIFIER'],axis=1)

	d.columns.name = 'ClientId'
	d.index.name = 'PathwaySo'

	return d.transpose(), None, indx_cols

'''
TODO
'''
def metabolomics_get_ClientId_lookup(indx_cols):
	ClientId_lookup = indx_cols.to_frame(index=False)
	new_cols = ClientId_lookup.iloc[0]
	new_cols[-1] = 'Group'
	ClientId_lookup.columns = new_cols
	ClientId_lookup
	ClientId_lookup = ClientId_lookup.drop(0).set_index('CLIENT IDENTIFIER')
	ClientId_lookup.index.name = 'ClientId'
	return ClientId_lookup 

'''
TODO
'''
def metabolomics_get_PathwaySo_lookup():
	skiprows = 0
	num_row_headers = 12
	num_col_headers = 17
	d_tmp = pd.read_csv('{}/{}'.format(
				data_dir
				, fname_metabolomics
				)
            , skiprows=metabolomics_skiprows
            , index_col=list(range(metabolomics_num_row_headers))
            , header=list(range(metabolomics_num_col_headers-1))
           )

	PathwaySo_lookup = d_tmp.index.to_frame(index=False)
	PathwaySo_lookup.columns = PathwaySo_lookup.iloc[0]
	PathwaySo_lookup = PathwaySo_lookup.drop(0).set_index("PATHWAY SORTORDER")
	PathwaySo_lookup.index.name = "PathwaySo"
	return PathwaySo_lookup

################################################################################
# Pairing samples in proteomics & metabolomics datasets
def omics_pairing_get_lookup():
	d = pd.read_csv('{}/{}'.format(data_dir, fname_omics_pairing)
			, usecols=list(range(omics_pairing_col_width))
		)
	# d = pd.read_csv(fname_omics_pairing, usecols=list(range(omics_pairing_col_width)))
	d= d[['Unique Sample ID \n(as labeled on tube)','Unique Sample ID \n(as labeled on tube).1']]
	d.columns = ['sampleId-flare', 'sampleId-remission']
	d = d.dropna().set_index('sampleId-flare', drop=False)
	return d