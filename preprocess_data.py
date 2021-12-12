import pandas as pd
import numpy as np
import os


def load_challenge_data(file):
    with open(file, 'r') as f:
        header = f.readline().strip()
        column_names = header.split('|')
        data = np.loadtxt(f, delimiter='|')

    # Ignore SepsisLabel column if present.
    if column_names[-1] == 'SepsisLabel':
        column_names = column_names[:-1]
        data = data[:, :-1]

    return data


# Specify the file directory where it is located
input_directory = "./input/"


# Find files.
files = []
for f in os.listdir(input_directory):
    if os.path.isfile(os.path.join(input_directory, f)) and not f.lower().startswith('.') and f.lower().endswith('psv'):
        files.append(f)


num_files = len(files)

for i, f in enumerate(files):
    print('    {}/{}...'.format(i+1, num_files))

    # Load data.
    input_file = os.path.join(input_directory, f)
    data = load_challenge_data(input_file)

#     # Make predictions.
#     num_rows = len(data)
#     scores = np.zeros(num_rows)
#     labels = np.zeros(num_rows)
#     for t in range(num_rows):
#         current_data = data[:t+1]
#         current_score, current_label = get_sepsis_score(current_data, model)
#         scores[t] = current_score
#         labels[t] = current_label
#
#     # Save results.
#     output_file = os.path.join(output_directory, f)
#     save_challenge_predictions(output_file, scores, labels)


# # Specify the file name where it is located
# file_name = "p000007.psv"
#
# input_file = file_path + file_name
# df = pd.read_csv(input_file, sep="|")          # skiprows=skip_rows

