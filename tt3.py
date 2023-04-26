import numpy as np

a = np.array([52854521, 14222760, 277279, 510247, 
             10238352,693393,560556,294590,250493,
             175663,166824,221048,188634,1067821,818622])

total = float(a.sum())

for i, num in enumerate(a):
    print("label: {}, inv freq: {:.4f}, inv sqrt freq: {:.4f}".format(i, 1 / (a[i] / total), 1 / np.sqrt(a[i]/total)))  