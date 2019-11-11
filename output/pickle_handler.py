import pickle

with open('embeddings.pickle', 'rb') as f:
    x = pickle.load(f)
    print (x)
