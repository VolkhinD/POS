import pickle
with open('./saved_files/results_dictionary.pkl', 'rb') as f:
    res = pickle.load(f)
    print(res['vanilla'])