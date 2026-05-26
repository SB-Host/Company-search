import pandas as pd
df = pd.read_csv('/Users/sawyerbrooks/company-search/cleaned_companies.csv', dtype=str).fillna('')
total = len(df)
complete = ((df['city_clean']!='') & (df['state_clean']!='') & (df['description']!='')).sum()
print(f'Complete:  {complete} / {total}')
print(f'Remaining: {total - complete}')
