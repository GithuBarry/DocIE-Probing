import os

import numpy as np

i = -1
X, Y = [], []
directory = os.getcwd()

while i < 1700:
    i += 1
    filename = "output_sentence" + str(i) + "_decoder_hidden_states.npy"
    f = os.path.join(directory, filename)
    if not os.path.isfile(f):
        continue

    x = np.load(f)
    x = np.array(x)
    print(x.shape)
    x = np.moveaxis(x, [1, 0], [0, 3])
    x = x[-1][0][0]
    x = x.flatten()
    appended = False

    x = np.append(x, (512 * 768 - len(x)) * [0])
    X.append(x)  # last layer, only batch
    if len(input[i]['triggers']) > 5:
        print()
    # Y.append(len(input[i]['triggers']))

X = np.array(X)
np.save("X-TANL_MucV1.npy")
# Y = np.array(Y)
