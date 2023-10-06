import os

dir_path = os.path.dirname(os.path.realpath(__file__))
file_path = os.path.join(dir_path, 'pr2table.json')

with open(file_path, 'r') as f:
    data = f.read()
print(data[:100])  # print the first 100 characters
