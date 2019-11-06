import pandas as pd
# d = {'col1': ['aa,bb', 'bb', 'cc,dd', 'aa,cc']}
# s = pd.DataFrame(d)
s = pd.read_csv(r'C:\Users\Administrator\PycharmProjects\Python1\mctagcsv.csv')
print(s)
separated_data = [el.split(',') for el in s['col1']]
print(separated_data)
keys = set([key for sublist in separated_data for key in sublist])
columns = {key: [1 if key in sublist else 0 for sublist in separated_data]
           for key in keys}
print(columns)
s1 = pd.DataFrame(columns)
print(s1)
s1.to_csv(r'C:\Users\Administrator\PycharmProjects\Python1\mctagcsvresult.csv')