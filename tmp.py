
def do_pca_on_phosphlow_data(data, plot_scree=True, plot_groups_1_2_only=True, n_components=30, show_plots=True):
    # number dimensions
    dims = data.shape[0] 
    # initialize and run SVD (that is PCA)
    svd = TruncatedSVD(n_components=n_components)
    svd.fit(data)
    eig_vals = svd.singular_values_**2
    
    if show_plots:
        plt.bar(np.arange(len(eig_vals))+1, eig_vals )
        plt.xlabel("eigenvalue value index");  plt.ylabel("eigenvalue")
        plt.title("scree plot of top eigenvalues")

    grp_1, grp_2, grp_3, grp_4 = (df.state == '01'),(df.state == '02'),(df.state == '03'),(df.state == '04')

    # Data has 32 dims; there are 54 samples
    PC_projection = svd.fit_transform(data)
    # PC_projection = np.dot(df_freq_centred, svd.components_.T)

    if show_plots:
        plt.figure(figsize=(10,10))
        s=500
        color=iter(plt.cm.flag(np.linspace(0,1,4)))
        plt.scatter(PC_projection[grp_1].T[0], PC_projection[grp_1].T[1], s=s, label='01')
        plt.scatter(PC_projection[grp_2].T[0], PC_projection[grp_2].T[1], s=s, label='02')
        if not plot_groups_1_2_only:
            plt.scatter(PC_projection[grp_3].T[0], PC_projection[grp_3].T[1], s=s, label='03')
            plt.scatter(PC_projection[grp_4].T[0], PC_projection[grp_4].T[1], s=s, label='04')
        plt.xlabel("PC1"); plt.ylabel("PC2")
        plt.legend()
    return svd.components_, PC_projection