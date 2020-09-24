import numpy as np
from sklearn import decomposition
from argparse import ArgumentParser
import scipy.sparse as sp


def pca(w, n_comp) -> np.ndarray:
    model = decomposition.PCA(n_comp)
    r = model.fit_transform(w)
    return r


if __name__ == '__main__':
    parser = ArgumentParser(description='')

    # Embedding
    parser.add_argument("--weights", required=True, type=str)

    args = parser.parse_args()

    if args.weights.endswith(".npz"):
        weights = sp.load_npz(args.weights)
        weights = weights.toarray()
    else:
        weights = np.load(args.weights)

    n = np.linspace(44, weights.shape[1], 10).astype(np.int)[:-1]
    
    for i in range(n.shape[0]):
        print(i)
        r = pca(weights, n[i])

        np.save(f"{args.weights}.pca-{n[i]}.npy", r)
