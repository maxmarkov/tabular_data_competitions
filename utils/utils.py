'''
Collection of utils for tabular data analysis.
Pretty chaotic now. To be unified in classes.
'''
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def correlation_matrix(df, columns, method='pearson', figsize=(15,10)):
    '''
    Method to compute the correlation matrix 
    '''
    # column names to indices 
    idx_cont = [df.columns.get_loc(c) for c in columns if c in df]
    # compute the correlation matrix for 'columns'
    c_matrix = df.iloc[:, idx_cont].corr(method=method)

    # Generate a mask for the upper triangle
    mask = np.zeros_like(c_matrix, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=figsize)

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(c_matrix, mask=mask, cmap=cmap, center=0, square=True, 
                linewidths=.5, cbar_kws={"shrink": .5}, annot=True)
    ax.set_title('Correlation matrix', fontsize=15)

def most_correlated_columns(df, columns, method='pearson', threshold=0.75):
    '''
    Method to find the most correlated features in the DataFrame
    '''
    # column names to indices 
    idx_cont = [df.columns.get_loc(c) for c in columns if c in df]
    # compute the correlation matrix for 'columns'
    c_matrix = df.iloc[:, idx_cont].corr(method=method)

    df_true = c_matrix[(c_matrix > threshold)]

    # find columns where the condition is valid
    columns = df_true.any(axis=0)
    columns = columns[columns==True]
    
    correlated_dict = {}
    for clmn in list(columns.index):
        row_index = list(df_true[clmn].dropna().index)
        value = list(df_true[clmn].dropna())
        temp = {}
        for i in range(len(row_index)):
            if row_index[i] == clmn:
                continue
            temp[row_index[i]] = round(value[i],3)
        if len(temp) > 0:
            correlated_dict[clmn] = temp
    return(correlated_dict)
