import pandas as pd

df = pd.read_csv('/Users/sawyerbrooks/company-search/cleaned_companies.csv', dtype=str).fillna('')

# Export rows that have a city AND description (state can be blank for non-US companies)
complete = df[(df['city_clean']!='') & (df['description']!='')].copy()

# Build Country column: USA if Country is USA or blank, otherwise use the country value
def get_country(row):
    c = str(row.get('Country', '')).strip()
    if c == '' or c == 'USA':
        return 'USA'
    return c

# Build City and State: blank them out for non-US companies (keep city if we have it)
def get_city(row):
    if get_country(row) == 'USA':
        return row.get('city_clean', '')
    return row.get('city_clean', '')  # keep city for international too

def get_state(row):
    if get_country(row) == 'USA':
        return row.get('state_clean', '')
    return ''  # blank state for non-US

complete['Country']   = complete.apply(get_country, axis=1)
complete['City_out']  = complete.apply(get_city, axis=1)
complete['State_out'] = complete.apply(get_state, axis=1)

export = complete[['Company', 'Category', 'City_out', 'State_out', 'Country', 'public_private', 'description', 'Website', 'LinkedIn', 'Employees', 'Year \nFounded']].copy()
export.columns = ['Company', 'Category', 'City', 'State', 'Country', 'Public/Private', 'Description', 'Website', 'LinkedIn', 'Employees', 'Year Founded']

output = '/Users/sawyerbrooks/Desktop/Companies_Enriched.xlsx'
export.to_excel(output, index=False)
print(f'Exported {len(export)} companies to {output}')
print(f'  US companies:            {(export["Country"]=="USA").sum()}')
print(f'  International companies: {(export["Country"]!="USA").sum()}')
