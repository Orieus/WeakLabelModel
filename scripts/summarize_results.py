#!/usr/bin/env python
import sys
from os import listdir
from os import path
from os import walk
from argparse import ArgumentParser

import csv
import pandas as pd

import matplotlib.pyplot as plt

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

    new_column_names = {0:'entry_n', 1:'subentry_n', 2:'date', 3:'time'}
    for i in range(5,df.shape[1],2):
        new_column_names[i] = df.ix[0,i-1]
    df.rename(columns=new_column_names, inplace=True)

    df.drop(list(range(4,df.shape[1],2)), axis=1, inplace=True)
    return df


def get_dataset_df(folder):
    filename = path.join(folder, 'dataset.csv')
    df = pd.read_csv(filename, header=None, quotechar='|',
            infer_datetime_format=True)
    df = format_diary_df(df)
    df.drop(['entry_n', 'subentry_n', 'date', 'time'], axis=1, inplace=True)
    return df


def get_results_df(folder):
    filename = path.join(folder, 'pd_df_results.csv')
    df = pd.read_csv(filename, quotechar='|', infer_datetime_format=True,
                     index_col=0)
    return df

def extract_summary(folder):
    dataset_df = get_dataset_df(folder)
    results_df = get_results_df(folder)

    dataset_df['folder'] = folder
    results_df['folder'] = folder

    summary = pd.merge(results_df, dataset_df)

    return summary


def extract_unfinished_summary(folder):
    dataset_df = get_dataset_df(folder)

    dataset_df['folder'] = folder

    return dataset_df


def main(folder='results'):
    results_folders, unfin_folders = get_list_results_folders(folder, True)

    u_summaries = []
    for uf in unfin_folders:
        u_summaries.append(extract_unfinished_summary(uf))
    dfs_unf = pd.concat(u_summaries, axis=0, ignore_index=True)

    print("The following experiments did not finish")
    print(dfs_unf)

    f_summaries = []
    for rf in results_folders:
        f_summaries.append(extract_unfinished_summary(rf))
    dfs_fin = pd.concat(f_summaries, axis=0, ignore_index=True)

    print("The following experiments did finish")
    print(dfs_fin)

    summaries = []
    for rf in results_folders:
        summaries.append(extract_summary(rf))

    dfs = pd.concat(summaries, axis=0, ignore_index=True)

    # from IPython import embed; embed()
    # idx = dfs.groupby(by=['folder'])['mean'].transform(min)==dfs['mean']

    # dfs[idx].to_csv('best_results.csv', sep='\t')

    groups_by = ['tag', 'name', 'method']
    columns = ['loss_train', 'loss_val']
    fig_extension = 'svg'
    for groupby in groups_by:
        for column in columns:
            # grouped = dfs[idx].groupby([groupby])
            grouped = dfs.groupby([groupby])

            dfs2 = pd.DataFrame({col:vals[column] for col,vals in grouped})
            meds = dfs2.median()
            meds.sort(ascending=False)
            dfs2 = dfs2[meds.index]

            fig = plt.figure(figsize=(10,len(meds)/2+3))
            ax = dfs2.boxplot(vert=False)
            ax.set_title('results grouped by {}'.format(groupby))

            counts =  {k:len(v) for k,v in grouped}
            ax.set_yticklabels(['%s\n$n$=%d'%(k,counts[k]) for k in meds.keys()])
            ax.set_xlabel(column)
            plt.tight_layout()
            fig.savefig('{}_{}.{}'.format(groupby, column, fig_extension))

def __test_1():
    main('results')
    sys.exit(0)


def parse_arguments():
    parser = ArgumentParser(description=("Generates a summary of all the " +
                                         "experiments in the subfolders of " +
                                         "the specified path"))
    parser.add_argument("results_path", metavar='PATH', type=str,
                        default='results',
                        help="Path with the result folders to summarize.")
    return parser.parse_args()


if __name__ == '__main__':
    __test_1()

    #args = parse_arguments()
    #main(args.results_path)
