import pandas
#read in the csv file with the name CrewData.csv as a dataframe
Crew_df = pandas.read_csv('CrewData.csv')
#see the first 5 rows of the data
print Crew_df.head()
#see the column heads of the data
print Crew_df.columns
#if I want to select the crew_ID column
ids=Crew_df.Crew_ID.values
#or
ids=Crew_df['Crew_ID'].values
#if I want to select seniority of pilot 900201 (first pilot)
#as the format of the Crew_ID column is string, so we use '900201'
pilot_one=Crew_df[Crew_df.Crew_ID == '900201']['Seniority'].values[0]
#if it is a number, then we will use 900201 directly
#pilot_one=Crew_df[Crew_df.Crew_ID == 900201]['Seniority'].values[0]
#read one row each time
for index,row in Crew_df.iterrows():
    print row['Crew_ID']
    print row['Seniority']