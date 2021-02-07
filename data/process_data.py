import sys
import pandas as pd 
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    '''
    INPUTS
    messages_filepath - filepath of the disaster_messages.csv
    categories_filepath - filepath of the disaster_categories.csv
    
    OUTPUT
    Returns the merged dataframe
    '''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages,categories,on='id')
    
    return df


def clean_data(df):
    """
    clean_data:
     1. Split Category Columns
     2. Convert Category Values to numeric
     3. Replace converted category values in df with new names
     4. Remove Duplicates
     5. Remove Unnecessary Rows and Columns
    
    Args:
    
    INPUT - df - merged Dataframe from load_data function
    OUTPUT - Returns cleaned Dataframe
    """
    
    # create a dataframe of the individuals category columns
    categories = df['categories'].str.split(';', expand=True)
    
    # Get new column names from category columns
    row = categories.iloc[0] # use this row to extract a list of new column names for categories.
    category_colnames = row.apply(lambda x: x.rstrip('- 0 1')) # up to the second to last character of each string with slicing
    categories.columns = category_colnames  # rename the columns of `categories`
    
    # Convert category values to 0 or 1
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]
    
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
    
    # drop the original categories column from `df`
    df.drop(['categories'],axis=1, inplace=True)
    
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df,categories], axis=1)
    
    #Drop duplicates from the dataframe
    df=df.drop_duplicates()
    
    
    # Find and remove rows that have different from "0" and "1" for each category column
    df_cat = df.iloc[:, 4:-1]
    for column in df_cat:
        indexnames = df_cat[(df_cat[column] != 0) & (df_cat[column] != 1)].index
        df.drop(indexnames, axis= 0, inplace=True)
    
    
    # Drop all the columns that only contain one unique value.
    def remove_single_unique_values(df):
        """
        Drop all the columns that only contain one unique value.
        not optimized for categorical features yet.
        """   
        cols_to_drop = df.nunique()
        cols_to_drop = cols_to_drop.loc[cols_to_drop.values==1].index
        df=df.drop(cols_to_drop, axis=1, inplace=True)
        return df
    remove_single_unique_values(df)
    
    
    return df
    
def save_data(df, database_filename, table_name='Disaster_messages'):
    """Save data into database.DisasterResponse.db
    Args:
        df: pandas.DataFrame. It contains disaster messages and categories that are cleaned.
        database_filename: String. Dataframe is saved into this database file.
        table_name: String. Dataframe is saved into this table on database.
    """
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql(table_name, engine, index=False, if_exists='replace', chunksize=600)
    
    
    
def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()