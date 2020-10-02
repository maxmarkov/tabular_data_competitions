'''
Collection of utils for tabular data analysis.
Pretty chaotic now. To be unified in classes.
'''
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class Correlation:
    '''
    Class to introduce the correlation matrix related methods
    Input: df - DataFrame with data,
           columns - list of selected columns for which the correlation matrix is computed 
           method - Method of correlation: {‘pearson’, ‘kendall’, ‘spearman’}
    '''
    def __init__(self, df, columns, method='pearson'):
        self.df = df
        self.columns = columns
        # all self.columns must be in df
        if not all(elem in self.df.columns for elem in list(self.columns)):
            raise ValueError ("Columns are not in the DataFrame")
        self.method = method

        # column names to indices
        idx_cont = [self.df.columns.get_loc(c) for c in self.columns if c in self.df]
        # compute the correlation matrix for 'columns'
        self.c_matrix = self.df.iloc[:, idx_cont].corr(method='pearson')

    def show_correlation_matrix(self, figsize=(15,10)):
        '''
        Method to show the correlation matrix in a nice graphical format using seaborn
        '''
    
        # Generate a mask for the upper triangle
        mask = np.zeros_like(self.c_matrix, dtype=np.bool)
        mask[np.triu_indices_from(mask)] = True
    
        # Set up the matplotlib figure
        fig, ax = plt.subplots(figsize=figsize)
    
        # Generate a custom diverging colormap
        cmap = sns.diverging_palette(220, 10, as_cmap=True)
    
        # Draw the heatmap with the mask and correct aspect ratio
        sns.heatmap(self.c_matrix, mask=mask, cmap=cmap, center=0, square=True, 
                    linewidths=.5, cbar_kws={"shrink": .5}, annot=True)
        ax.set_title('Correlation matrix', fontsize=15)

    def most_correlated_features(self, threshold=0.75):
        '''
        Method to find the most correlated features in the DataFrame
        Input: covariance threshold (float)
        Output: dictionary with feature covariance above threshold
        '''
        df = self.c_matrix[(self.c_matrix > threshold)]
    
        # find columns where the condition is valid
        columns = df.any(axis=0)
        columns = columns[columns==True]
       
        # create a dictionary of highly covariant features
        correlated_dict = {}
        for clmn in list(columns.index):
            row_index = list(df[clmn].dropna().index)
            value = list(df[clmn].dropna())
            temp = {}
            for i in range(len(row_index)):
                if row_index[i] == clmn:
                    continue
                temp[row_index[i]] = round(value[i],3)
            if len(temp) > 0:
                correlated_dict[clmn] = temp
        return(correlated_dict)

    def plot_feature_vs_target(self, feature, target):
        fig, axes = plt.subplots(1, 2, figsize=(15,5))

        # set ymax from the maximum of mean value
        ymax = self.df.groupby(feature)[target].mean().sort_values().max()
        print(ymax)

        self.df.plot.scatter(x=target, y=feature, style='*', s=10, c='r', ax=axes[0])
        axes[0].set_xlabel(target, fontsize=15)
        axes[0].set_ylabel(feature, fontsize=15)
        axes[0].set_ylim([0,ymax])
        #axes[0].set_xticklabels(fontsize=15)
        axes[0].set_title(feature+' vs '+ target + ' distribution', fontsize=15)

        self.df.groupby(feature)[target].mean().sort_values().plot(style='*', logx=True, grid=True, c='r', ax=axes[1])
        axes[1].set_xlabel(target, fontsize=15)
        axes[1].set_ylabel(feature, fontsize=15)
        axes[1].set_title(feature+' vs '+ target + ' mean', fontsize=15)

