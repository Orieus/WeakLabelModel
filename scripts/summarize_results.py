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

    h_size = max(7, len(columns)*.8)
    v_size = max(7, len(rows)*.8)
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
    best_tag = df.groupby('tag')['loss_val'].mean().argmin()
    mean_loss_best_tag = df[df.tag == best_tag][['name', 'loss_val']].groupby(by='name').mean().round(decimals=2)
    mean_loss_best_tag = mean_loss_best_tag.rename(columns={'loss_val': 'mean(superv_loss)'})
    df_table = pd.concat([df_table, mean_loss_best_tag], axis=1)

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


def wilcoxon_rank_sum_test(df, index, column, signed=False, twosided=True):
    indices = np.sort(df[index].unique())
    results = {}
    for i, index1 in enumerate(indices):
        results[index1] = df[df[index] == index1][column]
        # Ensure that the results are paired by dataset and simulation
        if i == 0:
            experiments = df[df[index] == index1][['sim', 'name']].values
        else:
            np.testing.assert_equal(experiments,
                                    df[df[index] == index1][['sim',
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
    results_folders, unfin_folders = get_list_results_folders(results_path, True)

    # Creates summary path if it does not exist
    if not os.path.exists(summary_path):
        os.mkdir(summary_path)

    u_summaries = []
    for uf in unfin_folders:
        u_summaries.append(extract_unfinished_summary(uf))

    if len(u_summaries) > 0 :
        dfs_unf = pd.concat(u_summaries, axis=0, ignore_index=True)

        print("The following experiments did not finish")
        print(dfs_unf)

    f_summaries = []
    for rf in results_folders:
        f_summaries.append(extract_unfinished_summary(rf))

    if len(f_summaries) == 0 :
        print("There are no finished experiments")
        return

    dfs_fin = pd.concat(f_summaries, axis=0, ignore_index=True)

    print("The following experiments did finish")
    print(dfs_fin)

    summaries = []
    for rf in results_folders:
        summaries.append(extract_summary(rf))

    df = pd.concat(summaries, axis=0, ignore_index=True)

    for key, value in filter_rows.items():
        if df[key].dtype == 'object':
            df = df[df[key].str.contains(value)]
        else:
            df = df[df[key] == float(value)]

    # Remove results with different parameters alpha and beta
    alpha = df['alpha'].unique()[0]
    df = df[df['alpha'] == alpha]
    beta = df['beta'].unique()[0]
    df = df[df['beta'] == beta]

    df.sort(columns=['date', 'time'], ascending=True, inplace=True)
    df.drop_duplicates(subset=['method', 'method2', 'n_classes', 'n_features',
                               'name', 'sim', 'size', 'tag'], inplace=True,
                       keep='last')

    df_datasets = export_datasets_info(df, path=summary_path)
    # Filter by best datasets
    best_datasets = df_datasets[df_datasets['mean(superv_loss)'] <= filter_performance].index
    df = df[df['name'].isin(best_datasets)]

    #friedman_test(df, 'tag', 'loss_val')
    print("Performing Wilcoxon rank sum test")
    df_wilc = wilcoxon_rank_sum_test(df, 'tag', 'loss_val')
    export_df(df_wilc, 'wilcoxon_rank_sum_test', path=summary_path, extension='pdf')
    df_wilc_square = df_wilc.pivot_table(index='index1', columns='index2',
                                         values='p-value')
    export_df(df_wilc_square, 'wilcoxon_rank_sum_test_square',
              path=summary_path, extension='pdf')
    fig = plot_df_heatmap(df_wilc_square,
                          title='Wilcoxon rank sum test p-values',
                          cmap=plt.cm.Greys)
    savefig_and_close(fig, 'wilcoxon_rank_sum_test_heatmap.{}'.format(
        fig_extension), path=summary_path)

    ########################################################################
    # Boxplots by different groups
    ########################################################################
    groups_by = ['tag', 'name', 'method']
    columns = ['loss_train', 'loss_val']
    for groupby in groups_by:
        for column in columns:
            # grouped = df[idx].groupby([groupby])
            grouped = df.groupby([groupby])

            df2 = pd.DataFrame({col: vals[column] for col, vals in grouped})
            meds = df2.median()
            meds.sort(ascending=False)
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
    # Heatmap of models vs method or dataset
    ########################################################################
    indices = ['method']
    columns = ['name']
    values = ['loss_val']
    normalizations = [None] #, 'rows', 'cols']
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
    # Heatmap of finished experiments name vs method
    ########################################################################
    df2 = pd.pivot_table(df, values='loss_val', index='name', columns='method',
                         aggfunc=len)
    df2 = df2.fillna(0).astype(int)
    fig = plot_df_heatmap(df2, title='Number of experiments',
                          cmap=plt.cm.Greys)
    savefig_and_close(fig,
                      'name_vs_method_heatmap_count.{}'.format(fig_extension),
                      path=summary_path)

    ########################################################################
    # Heatmap of models vs method or dataset
    ########################################################################
    indices = ['tag']
    columns = ['method', 'name']
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
                    df2 = pd.pivot_table(df, values=value, index=index,
                                         columns=column,
                                         aggfunc=np.mean)
                    title = 'Mean ER (a={}, b={}, {})'.format(
                            alpha, beta, norm)
                    fig = plot_df_heatmap(df2, normalize=norm, title=title)
                    savefig_and_close(fig, '{}_vs_{}_{}_heatmap_{}.{}'.format(
                                index, column, value, norm, fig_extension), path=summary_path)

    ########################################################################
    # Heatmap of models vs dataset for every method
    ########################################################################
    filter_by_column = 'method'
    filter_values = df[filter_by_column].unique()
    # TODO change columns and indices
    indices = ['tag']
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
                            fig = plot_df_heatmap(
                                    df2, normalize=norm,
                                    title='Heat-map by {} (normalized {})'.format(
                                        filtered_row, norm))
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
            filter_performance=0.7)
    main('results', summary_path='all', filter_rows={}, filter_performance=0.5)
    sys.exit(0)


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
    # TODO implement filter arguments
    main(**vars(args))
