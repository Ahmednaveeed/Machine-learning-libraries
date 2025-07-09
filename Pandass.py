"""
helps in data manipulation and analysis
handy when dealing with large datasets
has data structures like Series and DataFrame 
    series is a one-dimensional labeled array
    DataFrame is a two-dimensional labeled data structure with columns of potentially different types 
    and is similar to a spreadsheet or SQL table
# or a dictionary of Series objects
# can be used to read and write data in various formats like CSV, Excel, SQL databases, etc.
# can be used to perform data cleaning and preparation
# can be used to perform operations like filtering, grouping, merging, reshaping, and more
# can be used to handle missing data, perform statistical analysis, and visualize data
# can be used to perform time series analysis and manipulation
"""

import pandas as pd

df = pd.read_csv('pokemon_data.csv')
#print(df)               #print the entire DataFrame
#print(df.head(5))       #print the first 5 rows of the DataFrame

'''
df_xlsx = pd.read_excel('pokemon_data.xlsx')
print(df_xlsx.head(3))

df = pd.read_csv('pokemon_data.txt', delimiter='\t')
print(df.head(3))  
'''

'''
print(df.columns)        #print the column names of the DataFrame
print(df['Name'])        #print the 'Name' column of the DataFrame
print(df.iloc[1,4])      #print the value at row 1, column 4 (indexing starts at 0)
print(df.loc[1:4])       #print rows 1 to 4 (inclusive) of the DataFrame
print(df.describe())     #print the summary statistics of the DataFrame
print(df.loc[df['Type 1'] == 'Fire'])       #print rows where Type1 is 'Fire'
print(df.sort_values(['Type 1', 'HP'], ascending=[1, 1]).head(20))       #sort by Type1 and HP in ascending and descending order
'''

'''
df['Total'] = df['HP'] + df['Attack'] + df['Defense'] + df['Sp. Atk'] + df['Sp. Def'] + df['Speed']
print(df.head(5))       
df = df.drop(columns = ['Total'])
print(df.head(5))
df.to_csv('modified_data.csv')
'''

#print(df.loc[(df['Type 1'] == 'Fire') & (df['Type 2'] == 'Flying')])      #print rows where Type1 is 'Fire' and Type2 is 'Flying'
#print(df.loc[df['Name'].str.contains('Mega')])      #print rows where Name contains 'Mega'

'''
df.loc[df['Type 1'] == 'Fire', 'Type 1'] = 'Flamer'    #change Type1 from 'Fire' to 'Flamer'
print(df.head(20))       #print the first 5 rows of the DataFrame after the change

df.loc[df['Total'] > 500, 'Legendary', 'generation'] == ['Test1', 'Test 2']     #change Legendary and generation for rows where Total is greater than 500
print(df.head(40))
'''

#print(df.groupby(['Type 1']).size())
'''
print(                                  #group by Type1 and calculate mean of each group, sort by HP in descending order and print the first 10 rows
    df.groupby(['Type 1'])
    .mean()
    .sort_values('HP', ascending=False)
    .head(10)
    )        
'''

'''
for df in pd.read_csv('pokemon_data.csv', chunksize=5):  #read the CSV file in chunks of 5 rows
    print("DATA CHUNK:")
    print(df)  #print each chunk of 5 rows
'''