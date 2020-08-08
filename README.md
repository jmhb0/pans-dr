## pans-dr
Dimensionality reduction on biological data of patiens with [PANS](https://med.stanford.edu/pans.html). Functions for working with some data sets of mass cytometry, proteomics, and metabolomics that were shared for this analysis. This was a Summer 2020 research project at the [Bustamente lab](https://bustamantelab.stanford.edu/) at Stanford University.

There are two modules. One is for reading and structuring the data. One is for doing dimensionality reduction and prediction. The data is not included in the repo. The acutal analysis is also not in this repo, but are in notebooks that use these modules. This is to ensure data privacy. 

## Files
`get_data.py`: Declare filenames, and constants about the file structure for reading in `.csv`s. 

`data/`: Directory for csv files. The file names are declared in the `get_data` module. 

`omics_dr_tools.py`: Functions for finding mean-variance relationships of features; PCA; TSNE; UMAP; prediction using dim-reduced data. 


`.gitignore`: Ignore everything by default, and specify which files are allowed. This ensures no data is accidentally comitted (for example in `ipynb` cell outputs).

## Workflow
Create virtual environment and intall dependencies
```
virtualenv venv 
. venv/bin/activate
pip install -r requirements.txt
```

Add csv files to the `data/` directory and edit the filenames that are in the top 
of the file `get_data.py`. Create a notebook, `ipynb` file, and use the functions in `.py` files for anaylsis. 

## Analysis
The `ipynb` files that do the analysis are excluded from the repo. These are the tasks that would be done there:
- Read data using `get_data` module. 
- Explore dataset for outliers, do summary stats with `groupby()` funcs, etc. 
- Do data filtering. 
- Plot mean-variance relationship with funcs from `omics_dr_tools`, and do variance-stabalizing transformations based on that. 
- Do dimensionality reduction and prediction using funcs from `omics_dr_tools`. 

