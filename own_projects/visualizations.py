import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px  # interactive data visualization

from sklearn.decomposition import PCA


def get_kde_results(df, title, column1: str, xlabel1: str,
                    column2: str, xlabel2: str):
    """
    This function creates side-by-side visualizations of kernel density estimates.
        Parameters:
            df (data frame): data frame to work with
            title   (str): title of the visualizations
            column1 (str): first column to visualize
            xlabel1 (str): first plot's xlabel
            column2 (str): second column to visualize
            xlabel2 (str): second plot's xlabel
    """
    retained, quited = 0, 1

    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)  # create two subplots
    fig.set_figheight(5)
    fig.set_figwidth(15)  # set figure size
    fig.tight_layout(pad=3.0)  # add padding to separate figure edges
    fig.suptitle(title, fontsize=18)  # set title

    # create the first figure separated by classes
    sns.kdeplot(df[column1][(df['class'] == retained)],
                color='Red', shade=True, ax=ax1)
    sns.kdeplot(df[column1][(df['class'] == quited)],
                color='Blue', shade=True, ax=ax1)

    # create the second figure separated by classes
    sns.kdeplot(df[column2][(df['class'] == retained)],
                color='Red', shade=True, ax=ax2)
    sns.kdeplot(df[column2][(df['class'] == quited)],
                color='Blue', shade=True, ax=ax2)

    # set labels & legends
    fig.legend(['Retain', 'Churn'], loc='upper right')
    ax1.set_ylabel('Density')
    ax1.set_xlabel(xlabel1)

    ax2.set_ylabel('Density')
    ax2.set_xlabel(xlabel2)
    plt.show()


def get_pca_results(x, columns):
    """

    :param x:
    :param columns:
    :return:
    """
    pca = PCA().fit(x)
    exp_variance = pca.explained_variance_ratio_
    fig = px.line(x=np.arange(len(columns)), y=exp_variance, labels={'x': 'Features', 'y': 'PCA Results'})
    fig.update_layout(
        xaxis=dict(
            tickmode='array',
            tickvals=np.arange(len(columns)),
            ticktext=columns
        ),
        title_text='PCA Results'
    )
    fig.show()
