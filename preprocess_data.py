import numpy as np
import pandas as pd
import re

def split_df_list(df,target_column,separator):
    ''' df = dataframe to split,
    target_column = the column containing the values to split
    separator = the symbol used to perform the split
    returns: a dataframe with each entry for the target column separated, with each element moved into a new row. 
    The values in the other columns are duplicated across the newly divided rows.
    '''
    def split_list_rows(row,row_accumulator,target_column,separator):
        row[target_column] = re.sub(' +', ' ', row[target_column])
        split_row = row[target_column].split(separator)
        for s in split_row:
            new_row = row.to_dict()
            new_row[target_column] = s
            row_accumulator.append(new_row)
    new_rows = []
    df.apply(split_list_rows,axis=1,args = (new_rows,target_column,separator))
    new_df = pd.DataFrame(new_rows)
    return new_df


def read_from_uris(uri1, uri2=None, uri3=None):
    data_1 = pd.read_csv(uri1)
    data_2 = None
    data_3 = None
    if uri2 is not None:
        data_2 = pd.read_csv(uri2)
    if uri3 is not None:
        data_3 = pd.read_csv(uri3)
    if data_2 is not None and data_3 is not None:
        data = pd.concat([data_1, data_2, data_3], ignore_index = True)
    elif data_2 is not None:
        data = pd.concat([data_1, data_2], ignore_index = True)
        return data
    return data_1

def get_data(uri1, uri2=None, uri3=None):
	data = read_from_uris(uri1, uri2, uri3)
	split_df = split_df_list(data, 'content', '.')
	split_df = split_df[split_df['content'].str.len() > 0]
	split_df = split_df.drop(columns=['Unnamed: 0'])