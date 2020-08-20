import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import seaborn as sns
from collections import OrderedDict
import logging
import itertools 
import os
import json

from sklearn.linear_model import LinearRegression
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE
import umap

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import RandomizedSearchCV

from IPython.display import HTML

dir_path = os.path.dirname(os.path.realpath(__file__))

''' 
Constant to be used by Jupyter nb's. Creates a button/toggle
that hides the code boxes when viewing the HTML.

Example usage in a notebook
```
import omics_dr_tools as odr
from IPython.display import HTML
HTML(odr.HIDE_CODE_HTML)
```
'''
HIDE_CODE_HTML="""<script>
    code_show=true; 
    function code_toggle() {
     if (code_show){
     $('div.input').hide();
     } else {
     $('div.input').show();
     }
     code_show = !code_show
    } 
    $( document ).ready(code_toggle);
    </script>
    <form action="javascript:code_toggle()">
        <input type="submit" value="Click here to toggle on/off the raw code.">
    </form>"""
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=["b", "y", "c", "k"]) 

#------------------------------------------------------------------------------#
# Variance stabalizing 
''' 
Get 3 arrays: mean, vraiance, standard deviation. Each array has one entry per 
dimension (bio marker). 
This is called by `plot_mean_vs_var_and_std()`.

@param df: numeric dataframe/array with shape (n_samples,n_features)
returns: 3-tuple (mean,var,std), each with shape (n_features,1)
'''
def marker_stats(df):
    return df.mean(axis=0), df.var(axis=0), df.std(axis=0)

'''1d regression. If `fit_intercept` then y-intercept is must be zero '''
def regress_1d(X, y, fit_intercept=False):
    reg = LinearRegression(fit_intercept=fit_intercept, normalize=False)
    reg.fit(X, y)
    slope = reg.coef_[0]
    r_sq = reg.score(X, y)
    intercept = reg.intercept_
    return slope, r_sq, intercept

'''2 plots with mean on x axis. Plot 1 shows variance; plot 2 shows std.
Fit a linear model.'''
def plot_mean_vs_var_and_std(data
                             , fit_intercept=False
                             , xlim_0=None, ylim_0=None
                             , xlim_1=None, ylim_1=None
                             , s=10, figsize=(12,5)
                             , fig_title=''):
    
    mean_marker, var_marker, std_marker = marker_stats(data)
    mean_marker = np.reshape(np.asarray(mean_marker),(-1,1))

    f, axs = plt.subplots(1,2,figsize=figsize)
    f.suptitle(fig_title)
    
    axs[0].scatter(mean_marker, var_marker, s=s)
    slope, r_sq, intercept = regress_1d(mean_marker
                                        , var_marker
                                        , fit_intercept=fit_intercept)
    
    xlim = np.asarray(axs[0].get_xlim())
    axs[0].plot(xlim, xlim*slope+intercept, label="$R^2$={:.2f}".format(r_sq))
    axs[0].set(xlim=xlim)
    axs[0].legend()
    if xlim_0 is not None: axs[0].set(xlim=xlim_0)
    if ylim_0 is not None: axs[0].set(ylim=ylim_0)
    axs[0].set(xlabel='mean', ylabel='Variance')
    
    axs[1].scatter(mean_marker, std_marker, s=s)
    slope, r_sq, intercept = regress_1d(mean_marker
                                        , std_marker
                                        , fit_intercept=fit_intercept)
    xlim = np.asarray(axs[1].get_xlim())
    axs[1].plot(xlim, xlim*slope+intercept, label="$R^2$={:.2f}".format(r_sq))
    axs[1].set(xlim=xlim)
    axs[1].legend()    
    if xlim_1 is not None: axs[1].set(xlim=xlim_1)
    if ylim_1 is not None: axs[1].set(ylim=ylim_1)
    axs[1].set(xlabel='mean', ylabel='Std')
    return

def apply_common_VSTs(df, plot_name=''
                        , xlim_0=None, ylim_0=None
                        , xlim_1=None, ylim_1=None
                        ):
    # Iterate through common transforms
    transforms = [lambda df: np.arcsinh(df)
                  , lambda df: np.log(df+1)
                  , lambda df: df**0.5
                 ]
    transform_names = ['arcinh','log(x+1)','sqrt']
    assert len(transforms) == len(transform_names)

    for i, transform in enumerate(transforms):
        data = transform(df)
        plot_mean_vs_var_and_std(data
            , fit_intercept=True
            , fig_title="{}. Mean-var and mean-std after `{}` transform"\
                            .format(plot_name, transform_names[i])
            , xlim_0=xlim_0, ylim_0=ylim_0
            , xlim_1=xlim_1, ylim_1=ylim_1)

#------------------------------------------------------------------------------#
# Funcs for coloring dimensionally-reduced data
'''
Determine color for each data point that is being projected onto a low-dim space

Take a dataframe and a dimension on which to do coloring. Return list of colors
to be passed to axs.scatter(X,Y,color=color) as `color`. Also return `patches` 
object which can be added as a legend using axs.legend(patches=patches). This 
legend will be a list of categories for categorical data, or a colormap for 
continuous/gradient data.

@param data_labels: dataframe/series/array with shape (n_samples,). Data that 
    will define the coloring.
@param data_label_dtype: 'categorical' or 'continuous'. Whether the colors just 
    show discrete groups or whether we wish to plot a gradient

returns: 
    colors: list of colors to pass to axs.scatter(). Has shape (n_samples,1)
    patches: patches object to be passed to `axs.legend(patches=patches)`
    
NotImplemented: 
  The continuous case. This would require specifying a cmap and creating a 
mapping from values to colors so that we cover the full range. 
  When no. categories exceeds 7. This requires finding a categorical cmap with 
more colors. 
'''
def get_coloring_and_legend(data_labels, data_label_dtype='categorical'):
    if data_label_dtype=='continuous':
        raise NotImplementedError("TODO: implement continuous colormapping")
        
    elif data_label_dtype=='categorical':
        groups = np.sort(data_labels.unique())
        cnt_unique_labels = len(groups)
        
        # skip unlikely case there are many colors (would have to find a 
        # different, non-standard colormap list from mpl.cm.get_cmap())
        if cnt_unique_labels > 7:
            raise NotImplementedError(\
                "Must define a different colormap for more than 7 categories")

        # Use standard matplotib colors
        rgb_values = ['b', 'y', 'k','c', 'r', 'g', 'm']
        # fit to size
        rgb_values = rgb_values[:cnt_unique_labels]
        color_map = OrderedDict(zip(groups, rgb_values))

        # assign a coloring to each category 
        colors = data_labels.map(color_map)
        
        # manually create the legend patches
        patches = [mpatches.Patch(color=v, label=k) 
                        for k,v in color_map.items()] 
    else:
        raise ValueError()
    return colors, patches

#------------------------------------------------------------------------------#
'''
Use GROUPING_LABEL_MAP
@param data: a DataFrame
@ returns 
    data - filtered data 
    data_labels - mapped labels
'''
def get_data_groupings(data, ClientId_lookup, grouping, group_key='Group'):
    grouping = str(grouping)
    assert grouping in ("1","2","3","4","5","6","7")
    with open('{}/group-mappings.json'.format(dir_path)) as f:
        GROUPING_LABEL_MAP = json.load(f)
    drop_row_vals = GROUPING_LABEL_MAP[grouping]['drop']
    mapping = GROUPING_LABEL_MAP[grouping]['mapping']

    data_labels = ClientId_lookup.loc[data.index][group_key]
    data_labels = data_labels[~data_labels.isin(drop_row_vals)]
    data_labels = data_labels.map(mapping)
    
    return data.loc[data_labels.index], data_labels

#------------------------------------------------------------------------------#
# PCA 
'''
Centre and do PCA on a numeric dataframe. Optionally eigenvals, PC projections. 

Must pass in a dataframe so that we have index and column labels

This function calls `get_coloring_and_legend` which lives in this module.

Returns tuple of 5 objects. If you only want the first with name `PC+projection`
    `PC_projection, *args = do_pca(df)`
Or you can write 
    `*args, = do_pca(df)`
And then index it by args[0]
'''
def do_pca(data, data_labels=None, data_to_project=None
           , n_components=100
           , plot_eigenvalues=True, figsize_eig=(8,4)
           , plot_pc_projection=True, figsize_proj=(8,8), s=500
           , plot_pc_proj_X=1, plot_pc_proj_Y=2
           , data_label_dtype='categorical'
           , random_state=0
          ):
    # Centering
    shift = data.mean(axis=0)
    data = data - shift

    if data_to_project is None:
        data_to_project = data
    
    # PCA
    svd = TruncatedSVD(n_components=n_components, random_state=random_state)
    PC_projection = svd.fit(data).transform(data_to_project)
    PCs = svd.components_
    explained_variance_ratio = svd.explained_variance_ratio_
    eig_vals = svd.singular_values_**2
    
    # instantiate plotting objects to None (in case we don't enter `if` branch)
    f_eig = axs_eig = f_proj = ax_proj =None

    if plot_eigenvalues:
        f_eig, axs_eig = plt.subplots(1,1, figsize=figsize_eig)
        f_eig.suptitle("Scree plot of top {} eigenvalues"\
            .format(len(eig_vals)))
        axs_eig.bar(range(len(eig_vals)), eig_vals)
        
    if plot_pc_projection:
        # find which PC is to be plotted. Subtract 1 due to 0-indexing
        PC_X = PC_projection[:, plot_pc_proj_X-1]
        PC_Y = PC_projection[:, plot_pc_proj_Y-1]
        
        # Get coloring of data points based on 'SampleGroup'
        color, patches = get_coloring_and_legend(data_labels)
        
        # do the plotting
        f_proj, ax_proj = plt.subplots(1,1, figsize=figsize_proj)
        
        ax_proj.scatter(PC_X, PC_Y, s=s, color=color)
        ax_proj.set(xlabel="PC{}".format(plot_pc_proj_X)
               ,ylabel="PC{}".format(plot_pc_proj_Y))
        
        # manually create legend
        ax_proj.legend(handles=patches)
    
    
    ret_k = ['PC_projection', 'eig_vals', 'PCs', 'shift'
        , 'explained_variance_ratio', 'f_eig', 'axs_eig', 'f_proj', 'ax_proj']
    ret_v = [PC_projection, eig_vals, PCs, shift, explained_variance_ratio
                , f_eig, axs_eig, f_proj, ax_proj]
    return dict(zip(ret_k, ret_v))
    
#------------------------------------------------------------------------------#
''' 
Do TSNE and project to axis passed to labels
'''
def do_TSNE_PCA_reduced(data, data_labels, ax=None
    , data_label_dtype='categorical', random_state=0, perplexity=30
    , n_PCA_components=50, s=100, legend=True):
    r_do_pca  = do_pca(data, n_components=n_PCA_components
                                    , data_labels=data_labels
                                    , plot_eigenvalues=False
                                    , plot_pc_projection=False
                                   )
    PC_projection = r_do_pca['PC_projection']

    X_embedding = TSNE(n_components=2, perplexity=perplexity
                    ,random_state=random_state)\
                .fit_transform(PC_projection)
    
    # If no axis pased, create a simple figure for it
    if ax is None:
        f, ax = plt.subplots()    

    color, patches = get_coloring_and_legend(data_labels
        , data_label_dtype=data_label_dtype)
    ax.scatter(X_embedding[:,0], X_embedding[:,1], color=color,s=s)
    ax.set(title='Perplexity {}'.format(perplexity))
    if legend:    
        ax.legend(handles=patches)
    return X_embedding, ax

''' 
Run TSNE for a range of perplexities and plot on one figure.
'''
def do_TSNE_PCA_reduced_set_perplexities(data, data_labels,random_state=0
    , data_label_dtype='categorical', n_PCA_components=100, plot_width=10,s=100
    , legend=True, perplexities = [2,5,10,15,20,30,50,100]):
    figsize = (plot_width,2*plot_width)
    # ensure there are an even number of perplexity things 
    # this so we can reshape it to be in subplots
    if len(perplexities) % 2 == 1:
        perplexities.append(perplexity[-1])
    perplexities = np.reshape(perplexities, (-1,2))

    # Dict to hold embeddings. Will be returned.
    X_embeddings = {}

    f, axs = plt.subplots(*perplexities.shape, figsize=figsize)
    for i in range(perplexities.shape[0]):
        for j in range(perplexities.shape[1]):
            perplexity = perplexities[i,j]
            X_embedding, ax = do_TSNE_PCA_reduced(data, data_labels, ax=axs[i,j]
                , perplexity=perplexity, n_PCA_components=n_PCA_components
                , random_state=random_state, legend=legend)
            X_embeddings[perplexity] = X_embedding

    return X_embeddings, (f, axs)

#------------------------------------------------------------------------------#
def do_UMAP_PCA_reduced(data, data_labels, n_neighbors=10, ax=None
    , data_label_dtype='categorical', random_state=0, n_PCA_components=50
    , s=100, legend=True):
    r_do_pca  = do_pca(data, n_components=n_PCA_components
                                    , data_labels=data_labels
                                    , plot_eigenvalues=False
                                    , plot_pc_projection=False
                                   )

    PC_projection = r_do_pca['PC_projection']
    X_embedding = umap.UMAP(random_state=random_state
                    , n_neighbors=n_neighbors)\
                .fit_transform(PC_projection)

    # If no axis pased, create a simple figure for it
    if ax is None:
        f, ax = plt.subplots()    
    
    color, patches = get_coloring_and_legend(data_labels
        , data_label_dtype=data_label_dtype)
    ax.scatter(X_embedding[:,0], X_embedding[:,1], color=color,s=s)
    ax.set(title='n_neighbors {}'.format(n_neighbors))
    
    if legend:
        ax.legend(handles=patches)

    return X_embedding, ax
'''
Run UMAP for a range of n_neighbors and plot on one figure.
'''
def do_UMAP_PCA_reduced_set_n_neighbors(data, data_labels
    , data_label_dtype='categorical', n_PCA_components=50, plot_width=10,s=100
    , random_state=20, legend=True
    , n_neighbors = [2,3,4,5,6,7,8,10,12,15]):
    figsize = (plot_width,2*plot_width)
    # Ensure there are an even number of perplexity things 
    # This so we can reshape it to be in subplots
    if len(n_neighbors) % 2 == 1:
        n_neighbors.append(n_neighbors[-1])
    n_neighbors = np.reshape(n_neighbors, (-1,2))

    # Dict to hold embeddings. Will be returned.
    X_embeddings = {}

    f, axs = plt.subplots(*n_neighbors.shape, figsize=figsize)
    for i in range(n_neighbors.shape[0]):
        for j in range(n_neighbors.shape[1]):
            n = n_neighbors[i,j]
            X_embedding, ax = do_UMAP_PCA_reduced(data, data_labels,ax=axs[i,j]
                , n_neighbors=n, data_label_dtype='categorical', random_state=0
                , n_PCA_components=n_PCA_components, s=100, legend=True)
            X_embeddings[n] = X_embedding

    return X_embeddings, (f, axs)

#------------------------------------------------------------------------------#
'''
data must have index corresponding to sampleIds in `sample_pair_lookup`

`sample_pair_lookup` has index called with the sampleId of the flare patient
and a col called `sampleId-remission`.
'''
# Plotting lines between paired samples in an embedding 
def plot_pairs_on_dr_embedding(data, sample_pair_lookup, ax_proj
                        , lw=1, c='k'):
    scatter_points = ax_proj.collections[0].get_offsets().data
    df_scatter_points = pd.DataFrame(scatter_points, index=data.index)

    for flareId in sample_pair_lookup.index:
        remissionId = sample_pair_lookup.loc[flareId]['sampleId-remission']
        
        if flareId not in df_scatter_points.index:
            logging.warning("FlareId {} not in dataset".format(flareId))
            continue
        if remissionId not in df_scatter_points.index:
            logging.warning("RemissionId {} not in dataset".format(remissionId))
            continue
        x_pnts = df_scatter_points.loc[[flareId, remissionId]][0].values
        y_pnts = df_scatter_points.loc[[flareId, remissionId]][1].values
        ax_proj.plot(x_pnts, y_pnts, lw=lw, c=c)
    return
    # return f_proj, ax_proj

#------------------------------------------------------------------------------#
# Prediction code
'''
Simple random forest model that returns OOB scores, and detailed dataframe of 
results vs predictionss
'''
def random_forest_w_oob_scores(X, Y, **kwargs):
    # kwargs['oob_score'] = True
    clf = RandomForestClassifier(oob_score=True, **kwargs)
    clf.fit(X, Y)
    # get predictions
    oob_prediction_indx = np.argmax(clf.oob_decision_function_, axis=1)
    class_mapping = dict(zip(
        list(range(len(clf.classes_))), clf.classes_
    ))
    results = pd.DataFrame({'Y_test':Y, 'Y_pred': oob_prediction_indx})
    results['Y_pred'] = results['Y_pred'].map(class_mapping)
    return clf.oob_score_, results

''' 
Train a random forest model and make 1 prediction
@param X_train: training data, shape (n_samples, n_features)
@param Y_train: training data labels, shape (n_samples, 1)
@param X_test: a single test datum, shape (1,n_features)
@param kwargs: parameters that go to RandomForestClassifier in scikitlearn. 
They might be `random_state`, `n_estimators`, `max_depth`
returns: prediction of label for X_test, type string
'''
def predict_random_forest(X_train, Y_train, X_test, **kwargs):
    clf = RandomForestClassifier(**kwargs)
    clf.fit(X_train, Y_train.ravel())
    return clf.predict(X_test)[0] 

''' 
Given full set of data and labels, take a prediction model
@param X: all data features, shape (n_samples, n_features)
@param Y: all labels, shape (n_samples,1)
@param predict_model: prediction model with the form of 
`predict_random_forest`.
@param kwargs: keyword arguments specific to the model in `predict_model`. A 
common one might be `random_state` which ensures reproducibility across run.
@return: array `r` so that r[i,0] is the i'th sample and r[i,1] is the 
prediction for that sample when training `prediction_model` on all the
other data
'''
def leave_out_out_prediction(data, data_labels, predict_model
                    , n_PCA_components=50, **kwargs):
    X = np.asarray(data)
    Y = np.reshape(np.asarray(data_labels), (-1,1))

    assert X.shape[0]==Y.shape[0], \
        "data.shape[0] must equal data_labels.shape[0], the numnber of samples"

    m = X.shape[0]
    loo = LeaveOneOut()
    loo.get_n_splits(X)
    results = np.empty((m,2), dtype="<U10") 

    for train_index, test_index in loo.split(X):
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]
        
        # pass partial dataset to PCA, get projected data, project test data
        r_do_pca = do_pca(X_train
            , n_components=n_PCA_components
            , plot_eigenvalues=False, plot_pc_projection=False)
        # unpack return dictionary 
        X_train_projected, PCs, shift = (r_do_pca[k] 
                for k in ['PC_projection', 'PCs', 'shift'])

        X_test_projected = np.dot(PCs, (X_test-shift).transpose()).transpose()
        Y_predict = predict_model(X_train_projected, Y_train
                , X_test_projected, **kwargs)

        results[test_index[0]][0] = Y_test[0][0]
        results[test_index[0]][1] = Y_predict
        
    return pd.DataFrame(results, columns=['Y_test', 'Y_pred'])  

'''
Take in data in the form given by `leave_out_out_prediction`
and do summary statistics
'''
def get_confusion_matrix(results):
    labels = np.sort(results['Y_test'].unique())
    c_matrix = confusion_matrix(results['Y_test'], results['Y_pred']
                                , labels=labels)

    # confusion matrix standard + normalised
    c_matrix_df = pd.DataFrame(c_matrix, index=labels, columns=labels)
    c_matrix_df_norm = c_matrix_df.divide(
                            c_matrix_df.sum(axis=1)
                            , axis=0)
    return c_matrix_df, c_matrix_df_norm

'''
Plot confusion matrix for counts and normalised.
'''
def plot_confusion_matrix(results, figsize=(12,4), annotation_size=30
            , n_PCA_components=None, suptitle=None):
    c_matrix_df, c_matrix_df_norm = get_confusion_matrix(results)
    
    f, axs = plt.subplots(1,2,figsize=figsize)
    if n_PCA_components is not None:
        f.suptitle("Confusion matrix using top {} PCA components"\
            .format(n_PCA_components))
    if suptitle is not None:
        f.suptitle(suptitle)
    
    # For the counts matrix, just return the numbers but without the color
    # Different classes have different sizes, so the colors are deceptive. 
    # The colors in the normalized map make sense though
    sns.heatmap(c_matrix_df, ax=axs[0], annot=True
        , annot_kws={'size': annotation_size}
        ,cmap=ListedColormap(['white']),fmt='g'
        , linecolor='black', linewidths=1)

    axs[0].set(xlabel='Prediction', ylabel='True state'
        , title="Counts")
    
    # green colormap over range [0,1]
    sns.heatmap(c_matrix_df_norm, ax=axs[1], annot=True
        , annot_kws={'size': annotation_size}
        ,cmap='Greens',fmt='.2f', vmin=0, vmax=1)
    axs[1].set(xlabel='Prediction'
        , ylabel='True state', title="% correct")
    
    return f, axs


def tune_simple_rf_model_randomCV(data, data_labels, n_iter=30, cv=10, verbose=1
        , random_state=0):
    n_estimators_ = [100,500]
    max_features_ = list(np.arange(0, 1,0.2)+0.2)
    max_depth_ = [2,5,20,50,100,None]
    min_samples_split_ = [2,3,5]
    min_samples_leaf_ = [1,2,5]

    param_distributions = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
           }
    clf = RandomForestClassifier(oob_score=True)
    rf_random = RandomizedSearchCV(estimator = clf
        , param_distributions=param_distributions, n_iter=n_iter, cv=cv
        , verbose=verbose, random_state=random_state)
    # Fit the random search model
    rf_random.fit(data, data_labels)
    return rf_random.best_params_, rf_random.best_score_

'''
.
'''
def tune_simple_rf_w_exhaustiveOob(data, data_labels):
    n_estimators_ = [100,500]
    max_features_ = list(np.arange(0, 1,0.2)+0.2)
    max_depth_ = [2,5,20,50,100,None]
    # min_samples_split_ = [2,3,5]
    # min_samples_leaf_ = [1,2,5]

    param_distributions = {'n_estimators': n_estimators_,
                   'max_features': max_features_,
                   'max_depth': max_depth_,
                   # 'min_samples_split': min_samples_split_,
                   # 'min_samples_leaf': min_samples_leaf_,
           }
    p_keys = list(param_distributions.keys())
    p_values = param_distributions.values()

    # exhaustive grid search
    best_score = 0
    best_params = {}

    cnt = 1
    for val in itertools.product(*p_values):
        if cnt%100==0: print("cnt:{}, score:{:.2f}".format(cnt, best_score))
        cnt+=1
        # get parameters to try for this iteration
        params = {}
        for i in range(len(p_keys)):
            params[p_keys[i]] = val[i]

        clf = RandomForestClassifier(oob_score=True, random_state=0, **params)
        clf.fit(data, data_labels)
        score = clf.oob_score_
        if score > best_score:
            best_score = score
            best_params = params
    return best_params, best_score
