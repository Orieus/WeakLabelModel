#!/usr/bin/env python
import sys
import itertools
import os
from os import walk
from argparse import ArgumentParser

import json
import pandas as pd
import numpy as np
from scipy.stats import friedmanchisquare
from scipy.stats import wilcoxon, ranksums
import cairocffi as cairo
from weasyprint import HTML

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig_extension = 'svg'


def savefig_and_close(fig, figname, path='', bbox_extra_artists=None):
    filename = os.path.join(path, figname)
    fig.savefig(filename, bbox_extra_artists=bbox_extra_artists,
                bbox_inches='tight')
    fig.clear()
    plt.close(fig)


class MyFloat(float):
    def _remove_leading_zero(self, value, string):
        if 1 > value > -1:
            string = string.replace('0', '', 1)
        return string

    def __str__(self):
        string = super(MyFloat, self).__str__()
        return self._remove_leading_zero(self, string)

    def __format__(self, format_string):
        string = super(MyFloat, self).__format__(format_string)
        return self._remove_leading_zero(self, string)


def plot_df_heatmap(df, normalize=None, title='Heat-map',
                    cmap=plt.cm.Blues, colorbar=False):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.

    normalize : 'rows', 'cols' (default=None)
    """
    rows = df.index.values
    columns = df.columns.values
    M = df.values

    if normalize == 'rows':
        M = M.astype('float') / M.sum(axis=1)[:, np.newaxis]
    if normalize == 'cols':
        M = M.astype('float') / M.sum(axis=0)[np.newaxis, :]

    h_size = len(columns)*.5 + 2
    v_size = len(rows)*.5 + 2
    fig = plt.figure(figsize=(h_size, v_size))
    ax = fig.add_subplot(111)
    im = ax.imshow(M, interpolation='nearest', cmap=cmap)
    if colorbar:
        fig.colorbar(im)
    ax.set_title(title)
    column_tick_marks = np.arange(len(columns))
    ax.set_xticks(column_tick_marks)
    ax.set_xticklabels(columns, rotation=45, ha='right')
    row_tick_marks = np.arange(len(rows))
    ax.set_yticks(row_tick_marks)
    ax.set_yticklabels(rows)

    thresh = np.nanmin(M) + ((np.nanmax(M)-np.nanmin(M)) / 2.)
    are_ints = df.dtypes[0] in ['int', 'int32', 'int64']
    for i, j in itertools.product(range(M.shape[0]), range(M.shape[1])):
        # fontsize is adjusted for different number of digits
        if are_ints:
            ax.text(j, i, M[i, j], horizontalalignment="center",
                    verticalalignment="center", color="white" if M[i, j] >
                    thresh else "black")
        else:
            if np.isfinite(M[i, j]):
                ax.text(j, i, '{:0.2f}'.format(MyFloat(M[i, j])),
                        horizontalalignment="center",
                        verticalalignment="center",
                        color="white" if M[i, j] > thresh else "black")

    ax.set_ylabel(df.index.name)
    ax.set_xlabel(df.columns.name)
    fig.tight_layout()
    return fig


def get_list_results_folders(folder, return_unfinished=False):
    essentials = ['description.txt', 'dataset.csv']
    finished = ['pd_df_results.csv']
    list_results_folders = []
    list_unfinished_folders = []
    for root, subdirs, files in walk(folder):
        if set(essentials).issubset(set(files)):
            if set(finished).issubset(set(files)):
                list_results_folders.append(root)
            elif return_unfinished:
                list_unfinished_folders.append(root)

    if return_unfinished:
        return list_results_folders, list_unfinished_folders

    return list_results_folders


def format_diary_df(df):
    df[2] = pd.to_datetime(df[2])
    df[3] = pd.to_timedelta(df[3], unit='s')

    new_column_names = {0: 'entry_n', 1: 'subentry_n', 2: 'date', 3: 'time'}
    for i in range(5, df.shape[1], 2):
        new_column_names[i] = df.ix[0, i-1]
    df.rename(columns=new_column_names, inplace=True)

    df.drop(list(range(4, df.shape[1], 2)), axis=1, inplace=True)
    return df


def get_dataset_df(folder, keep_time=False):
    filename = os.path.join(folder, 'dataset.csv')
    df = pd.read_csv(filename, header=None, quotechar='|',
                     infer_datetime_format=True)
    df = format_diary_df(df)
    if keep_time:
        to_drop = ['entry_n', 'subentry_n']
    else:
        to_drop = ['entry_n', 'subentry_n', 'date', 'time']
    df.drop(to_drop, axis=1, inplace=True)
    return df


def get_results_df(folder):
    filename = os.path.join(folder, 'pd_df_results.csv')
    df = pd.read_csv(filename, quotechar='|', infer_datetime_format=True,
                     index_col=0)
    return df


def extract_summary(folder):
    dataset_df = get_dataset_df(folder, keep_time=True)
    results_df = get_results_df(folder)

    dataset_df['folder'] = folder
    results_df['folder'] = folder

    summary = pd.merge(results_df, dataset_df)

    return summary


def extract_unfinished_summary(folder):
    dataset_df = get_dataset_df(folder)

    dataset_df['folder'] = folder

    return dataset_df


def export_datasets_info(df, path='', stylesheets=['style.css']):
    columns = ['name', 'size', 'n_features', 'n_classes', 'seconds']
    sort_by = ['name']
    index = columns[0]
    df_by_seconds = df[columns].sort_values('seconds', ascending=False)
    df_table = df_by_seconds.drop_duplicates(subset=columns[:-1],
                                             keep='first'
                                             ).sort_values(sort_by
                                                           ).set_index(index)
    # Export to LaTeX
    df_table.drop('seconds', axis=1).to_latex(os.path.join(path,
                                                           "datasets.tex"))

    # Add mean performance of best model
    best_model = df.groupby('model')['loss_val'].mean().argmin()
    mean_loss_best_model = df[df.model == best_model][['name', 'loss_val']].groupby(by='name').mean().round(decimals=2)
    mean_loss_best_model = mean_loss_best_model.rename(columns={'loss_val': 'mean(superv_loss)'})
    df_table = pd.concat([df_table, mean_loss_best_model], axis=1)

    # Export to pdf
    html_out = df_table.to_html()
    filename = os.path.join(path, "datasets.pdf")
    HTML(string=html_out).write_pdf(filename, stylesheets=stylesheets)

    return df_table


def export_df(df, filename, path='', extension='pdf',
              stylesheets=['style.css']):
    if extension == 'pdf':
        html_out = df.to_html()
        filename = os.path.join(path, filename)
        HTML(string=html_out).write_pdf("{}.{}".format(filename, extension),
                                        stylesheets=stylesheets)


def friedman_test(df, index, column):
    indices = np.sort(df[index].unique())

    results = {}
    first = True
    for ind in indices:
        results[ind] = df[df[index] == ind][column].values
        if first:
            size = results[ind].shape[0]
            first = False
        elif size != results[ind].shape[0]:
            print("Friedman test can not be done with different sample sizes")
            return

    # FIXME be sure that the order is correct
    statistic, pvalue = friedmanchisquare(*results.values())
    return statistic, pvalue


def wilcoxon_rank_sum_test(df, index, column, signed=False,
                           twosided=True):
    indices = np.sort(df[index].unique())
    results = {}
    for i, index1 in enumerate(indices):
        results[index1] = df[df[index] == index1][column]
        # Ensures that all the results are aligned by simulation number (column
        # sim), dataset (column name) and mixing_matrix_M
        if i == 0:
            experiments = df[df[index] == index1][['sim', 'mixing_matrix_M',
                                                   'name']].values
        else:
            np.testing.assert_equal(experiments,
                                    df[df[index] == index1][['sim',
                                                             'mixing_matrix_M',
                                                             'name']].values)
    stat = []
    for (index1, index2) in itertools.combinations(indices, 2):
        if index1 != index2:
            if signed:
                statistic, pvalue = wilcoxon(results[index1].values,
                                             results[index2].values)
            else:
                statistic, pvalue = ranksums(results[index1].values,
                                             results[index2].values)
            if not twosided:
                pvalue /= 2
            stat.append(pd.DataFrame([[index1, index2, statistic, pvalue]],
                                     columns=['index1', 'index2', 'statistic',
                                              'p-value']))

    dfstat = pd.concat(stat, axis=0, ignore_index=True)
    return dfstat


def main(results_path='results', summary_path='', filter_rows={},
         filter_performance=1.0):
    print('\n#########################################################')
    print('##### Making summary of folder {}'.format(results_path))
    print('#')
    results_folders, unfin_folders = get_list_results_folders(results_path, True)

    # Creates summary path if it does not exist
    if not os.path.exists(summary_path):
        os.mkdir(summary_path)

    u_summaries = []
    for uf in unfin_folders:
        u_summaries.append(extract_unfinished_summary(uf))

    if len(u_summaries) > 0 :
        dfs_unf = pd.concat(u_summaries, axis=0, ignore_index=True)

        print("Experiments that did not finish: {}".format(len(u_summaries)))

    f_summaries = []
    for rf in results_folders:
        f_summaries.append(extract_unfinished_summary(rf))

    if len(f_summaries) == 0 :
        print("There are no finished experiments")
        return

    dfs_fin = pd.concat(f_summaries, axis=0, ignore_index=True)

    print("Experiments finished: {}".format(len(f_summaries)))

    summaries = []
    for rf in results_folders:
        summaries.append(extract_summary(rf))

    df = pd.concat(summaries, axis=0, ignore_index=True)

    print("Filtering only rows that contain")
    for key, value in filter_rows.items():
        print("- [{}] = {}".format(key, value))
        if df[key].dtype == 'object':
            df = df[df[key].str.contains(value)]
        else:
            df = df[df[key] == float(value)]

    # Remove results with different parameters alpha and beta
    alpha = df['alpha'].unique()[0]
    df = df[df['alpha'] == alpha]
    beta = df['beta'].unique()[0]
    df = df[df['beta'] == beta]

    # Remove synthetic datasets
    df = df[~df['name'].str.contains('blobs|gauss_quantiles')]

    print("Analyzing experiments with alpha={} & beta={}".format(alpha, beta))
    df.rename(columns={'tag': 'model', 'method': 'mixing_matrix_M'},
              inplace=True)

    df.replace({'model': {'Keras-LR-Superv-SGD': '(1) LR-Superv',
                          'Keras-LR-Mproper-SGD': '(2) LR-Mproper',
                          'Keras-LR-Weak-SGD': '(3) LR-Weak',
                          'Keras-LR-VLL-quasi_IPL-SGD': '(4) LR-qIPL',
                          'Keras-LR-OSL-SGD': '(5) LR-OSL',
                          'Keras-MLP-Superv-SGD': '(1) FNN-Superv',
                          'Keras-MLP-Mproper-SGD': '(2) FNN-Mproper',
                          'Keras-MLP-Weak-SGD': '(3) FNN-Weak',
                          'Keras-MLP-VLL-quasi_IPL-SGD': '(4) FNN-qIPL',
                          'Keras-MLP-OSL-SGD': '(5) FNN-OSL'}}, inplace=True)

    df.sort_values(by=['date', 'time'], ascending=True, inplace=True)
    df.drop_duplicates(subset=['mixing_matrix_M', 'method2', 'n_classes',
                               'n_features', 'name', 'sim', 'size', 'model'],
                       inplace=True,
                       keep='last')

    df_datasets = export_datasets_info(df, path=summary_path)
    # Filter by best datasets
    best_datasets = df_datasets[df_datasets['mean(superv_loss)'] <= filter_performance].index
    df = df[df['name'].isin(best_datasets)]

    #friedman_test(df, 'model', 'loss_val')
    print("Performing Wilcoxon rank sum test")
    mixing_matrices_M = df['mixing_matrix_M'].unique()
    for mmM in mixing_matrices_M:
        print("- Mixing matrix M is {}".format(mmM))
        df_wilc = wilcoxon_rank_sum_test(df[df['mixing_matrix_M'] == mmM],
                                         'model', 'loss_val', twosided=False)
        export_df(df_wilc, 'wilc_r_s_test_{}'.format(mmM),
                  path=summary_path, extension='pdf')
        df_wilc_square = df_wilc.pivot_table(index='index1', columns='index2',
                                             values='p-value')
        export_df(df_wilc_square,
                  'wilc_r_s_test_square_{}'.format(mmM),
                  path=summary_path, extension='pdf')
        fig = plot_df_heatmap(df_wilc_square,
                              title='W-r-s-t p-v {}'.format(mmM),
                              cmap=plt.cm.Greys)
        savefig_and_close(fig, 'wilc_r_s_test_{}.{}'.format(
                          mmM, fig_extension), path=summary_path)

    mmM = 'all'
    print("- Mixing matrix M is {}".format(mmM))
    df_wilc = wilcoxon_rank_sum_test(df, 'model', 'loss_val', twosided=False)
    export_df(df_wilc, 'wilc_r_s_test_{}'.format(mmM),
              path=summary_path, extension='pdf')
    df_wilc_square = df_wilc.pivot_table(index='index1', columns='index2',
                                         values='p-value')
    export_df(df_wilc_square,
              'wilc_r_s_test_square_{}'.format(mmM),
              path=summary_path, extension='pdf')
    fig = plot_df_heatmap(df_wilc_square,
                          title='W-r-s-t p-v {}'.format(mmM),
                          cmap=plt.cm.Greys)
    savefig_and_close(fig, 'wilc_r_s_test_{}.{}'.format(
                      mmM, fig_extension), path=summary_path)

    ########################################################################
    # Boxplots by different groups
    ########################################################################
    groups_by = ['model', 'name', 'mixing_matrix_M']
    columns = ['loss_train', 'loss_val']
    for groupby in groups_by:
        for column in columns:
            # grouped = df[idx].groupby([groupby])
            grouped = df.groupby([groupby])

            df2 = pd.DataFrame({col: vals[column] for col, vals in grouped})
            meds = df2.median()
            meds.sort_values(ascending=False, inplace=True)
            df2 = df2[meds.index]

            fig = plt.figure(figsize=(10, len(meds)/2+3))
            ax = df2.boxplot(vert=False)
            ax.set_title('results grouped by {}'.format(groupby))

            counts =  {k: len(v) for k, v in grouped}
            ax.set_yticklabels(['%s\n$n$=%d' % (k, counts[k]) for k in meds.keys()])
            ax.set_xlabel(column)
            savefig_and_close(fig, '{}_{}.{}'.format(groupby, column,
                                                     fig_extension), path=summary_path)

    ########################################################################
    # Heatmap of models vs mixing_matrix_M or dataset
    ########################################################################
    indices = ['mixing_matrix_M']
    columns = ['name']
    values = ['loss_val']
    for value in values:
        for index in indices:
            for column in columns:
                df2 = pd.pivot_table(df, values=value, index=index,
                                     columns=column,
                                     aggfunc=len)
                fig = plot_df_heatmap(df2, title='Number of experiments',
                                      cmap=plt.cm.Greys)
                savefig_and_close(fig, '{}_vs_{}_{}_heatmap_count.{}'.format(
                            index, column, value, fig_extension), path=summary_path)


    ########################################################################
    # Heatmap of finished experiments name vs mixing_matrix_M
    ########################################################################
    df2 = pd.pivot_table(df, values='loss_val', index='name', columns='mixing_matrix_M',
                         aggfunc=len)
    df2 = df2.fillna(0).astype(int)
    fig = plot_df_heatmap(df2, title='Number of experiments',
                          cmap=plt.cm.Greys)
    savefig_and_close(fig, 'name_vs_mixing_matrix_M_heatmap_count.{}'.format(
                        fig_extension), path=summary_path)

    ########################################################################
    # Heatmap of models vs mixing_matrix_M or dataset
    ########################################################################
    indices = ['model']
    columns = ['mixing_matrix_M', 'name']
    values = ['loss_val']
    normalizations = [None] #, 'rows', 'cols']
    for value in values:
        for index in indices:
            for column in columns:
                df2 = pd.pivot_table(df, values=value, index=index,
                        columns=column, aggfunc=len, fill_value=0).astype(int)
                fig = plot_df_heatmap(df2, title='Number of experiments',
                                      cmap=plt.cm.Greys)
                savefig_and_close(fig, '{}_vs_{}_{}_heatmap_count.{}'.format(
                            index, column, value, fig_extension), path=summary_path)
                for norm in normalizations:
                    # Mean error rates
                    df2 = pd.pivot_table(df, values=value, index=index,
                                         columns=column,
                                         aggfunc=np.mean)
                    title = r'Mean ER ($\alpha={}$, $\beta={}$)'.format(
                            alpha, beta, norm)
                    fig = plot_df_heatmap(df2, normalize=norm, title=title)
                    savefig_and_close(fig, '{}_vs_{}_{}_heatmap_{}.{}'.format(
                                index, column, value, norm, fig_extension), path=summary_path)

                    # Median error rates
                    df2 = pd.pivot_table(df, values=value, index=index,
                                         columns=column,
                                         aggfunc=np.median)
                    title = r'Median ER ($\alpha={}$, $\beta={}$)'.format(
                            alpha, beta, norm)
                    fig = plot_df_heatmap(df2, normalize=norm, title=title)
                    savefig_and_close(fig, '{}_vs_{}_{}_median_heatmap_{}.{}'.format(
                                index, column, value, norm, fig_extension), path=summary_path)

    ########################################################################
    # Heatmap of models vs dataset for every mixing_matrix_M
    ########################################################################
    filter_by_column = 'mixing_matrix_M'
    filter_values = df[filter_by_column].unique()
    # TODO change columns and indices
    indices = ['model']
    columns = ['name', 'size', 'n_classes', 'n_features']
    values = ['loss_val']
    normalizations = [None] #, 'rows', 'cols']
    for filtered_row in filter_values:
        for value in values:
            for index in indices:
                for column in columns:
                    df_filtered = df[df[filter_by_column] == filtered_row]
                    df2 = pd.pivot_table(df_filtered, values=value,
                                         index=index, columns=column)
                    if df2.columns.dtype in ['object', 'string']:
                        for norm in normalizations:
                            title = r'Heat-map by {} ($\alpha={}$, $\beta={}$)'.format(
                                        filtered_row, alpha, beta)
                            fig = plot_df_heatmap(df2, normalize=norm, title=title)
                            savefig_and_close(fig, '{}_vs_{}_by_{}_{}_heatmap_{}.{}'.format(
                                        index, column, filtered_row, value, norm,
                                        fig_extension), path=summary_path)

                        df2 = pd.pivot_table(df, values=value, index=index,
                                             columns=column, aggfunc=len,
                                             fill_value=0).astype(int)
                        fig = plot_df_heatmap(df2, title='Number of experiments',
                                              cmap=plt.cm.Greys)
                        savefig_and_close(fig, '{}_vs_{}_by_{}_{}_heatmap_count.{}'.format(
                                    index, column, filtered_row, value, fig_extension), path=summary_path)
                    else:
                        fig = plt.figure()
                        ax = fig.add_subplot(111)
                        df2.transpose().plot(ax=ax, style='.-', logx=True)
                        ax.set_title('{} {}'.format( filtered_row, value))
                        ax.set_ylabel(value)
                        lgd = ax.legend(loc='center left', bbox_to_anchor=(1, .5))
                        savefig_and_close(fig,
                                'plot_mean_{}_vs_{}_by_{}_{}.{}'.format( index,
                                    column, filtered_row, value,
                                    fig_extension), path=summary_path, bbox_extra_artists=(lgd,))


def __test_1():
    main('results', summary_path='keras_lr', filter_rows={'tag':'Keras-LR'},
            filter_performance=0.7)
    main('results', summary_path='keras_mlp', filter_rows={'tag':'Keras-MLP'},
            filter_performance=1.0)
    main('results', summary_path='all', filter_rows={}, filter_performance=0.5)
    sys.exit(0)

def __test_2():
    main('../results_a7_b3', summary_path='test_1', filter_rows={'tag':'Keras-MLP'},
         filter_performance=1.0)

def parse_arguments():
    parser = ArgumentParser(description=("Generates a summary of all the " +
                                         "experiments in the subfolders of " +
                                         "the specified path"))
    parser.add_argument("results_path", metavar='PATH', type=str,
                        default='results',
                        help="Path with the result folders to summarize.")
    parser.add_argument("summary_path", metavar='SUMMARY', type=str,
                        default='',
                        help="Path to store the summary.")
    parser.add_argument("-p", "--performance", metavar='PERFORMANCE',
                        type=float, default=1.0, dest='filter_performance',
                        help="Filter all the datasets with supervised performance lower than the specified value")
    parser.add_argument("-f", "--filter", type=json.loads,
                        default='{}', dest='filter_rows',
                        help="Dictionary with columns and filters.")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    main(**vars(args))
