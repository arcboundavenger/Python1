import pandas as pd
s = pd.read_csv('mctagcsv2.csv')
separated_data = [el.split(',') for el in s['col1']]
keys = set([key for sublist in separated_data for key in sublist])
print('keys:')
print(keys)
columns = {key: [1 if key in sublist else 0 for sublist in separated_data]
           for key in keys}
print(columns)
s1 = pd.DataFrame(columns)
print(s1)
s1.to_csv('mctagcsvresult3.csv')