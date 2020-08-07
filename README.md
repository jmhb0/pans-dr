## pans-dr
Dimensionality reduction on biological data of patiens with [PANS](https://med.stanford.edu/pans.html). Functions for working with some specific mass cytometry, proteomics, and metabolomics data sets that were shared for this analysis. This was a Summer 2020 research project at the [Bustamente lab](https://bustamantelab.stanford.edu/) at Stanford University.

There are two modules. One is for reading and structuring the data. One is for doing dimensionality reduction and prediction. The data and the analysis itself is not in the repo but was done in notebooks that use the modules in this repo. This is to ensure data privacy. 

## Repo structure
`get_data.py`: Declare filenames, and constants about the file structure for reading in `.csv`s. 

`omics_dr_tools.py`: Functions for finding mean-variance relationships of features; PCA anlysis; TSNE; UMAP; prediction using dim-reduced data. 

`data/`: Hold's csv files. The file names are declared in the `get_data` module. 

`.gitignore`: Ignore everything by default, and specify which files are allowed. This ensures no data is accidentally comitted, for example by saving `ipynb` cell outputs. 

## Workflow
Create virtual environment and intall dependencies
```
virtualenv venv 
. venv/bin.activate
pip install -r requirements.txt
```

Add csv files to the `data/` directory and edit the filenames that are in the top 
of the file `get_data.py`. Create a notebook, `ipynb` file, and use the functions in `.py` files for anaylsis. 

## Analsis
The `ipynb` files that do the analysis are excluded from the repo. These are the tasks that would be done there:
- Read data using `get_data` module. 
- Explore dataset for outliers, do summary stats with `groupby()` funcs, etc. 
- Do data filtering. 
- Plot mean-variance relationship with funcs from `omics_dr_tools`, and do variance-stabalizing transformations based on that. 
- Do dimensionality reduction and prediction using funcs from `omics_dr_tools`. 

