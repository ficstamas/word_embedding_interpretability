import numpy as np
from argparse import ArgumentParser


def whiten(X: np.ndarray, method='zca'):
    """
    Source: https://gist.github.com/joelouismarino/ce239b5601fff2698895f48003f7464b
    Whitens the input matrix X using specified whitening method.
    Inputs:
        X:      Input data matrix with data examples along the first dimension
        method: Whitening method. Must be one of 'zca', 'zca_cor', 'pca',
                'pca_cor', or 'cholesky'.
    """
    X = X.reshape((-1, np.prod(X.shape[1:])))
    X_centered = X - np.mean(X, axis=0)
    Sigma = np.dot(X_centered.T, X_centered) / X_centered.shape[0]
    W = None

    if method in ['zca', 'pca', 'cholesky']:
        U, Lambda, _ = np.linalg.svd(Sigma)
        if method == 'zca':
            W = np.dot(U, np.dot(np.diag(1.0 / np.sqrt(Lambda + 1e-5)), U.T))
        elif method == 'pca':
            W = np.dot(np.diag(1.0 / np.sqrt(Lambda + 1e-5)), U.T)
        elif method == 'cholesky':
            W = np.linalg.cholesky(np.dot(U, np.dot(np.diag(1.0 / (Lambda + 1e-5)), U.T))).T
    elif method in ['zca_cor', 'pca_cor']:
        V_sqrt = np.diag(np.std(X, axis=0))
        P = np.dot(np.dot(np.linalg.inv(V_sqrt), Sigma), np.linalg.inv(V_sqrt))
        G, Theta, _ = np.linalg.svd(P)
        if method == 'zca_cor':
            W = np.dot(np.dot(G, np.dot(np.diag(1.0 / np.sqrt(Theta + 1e-5)), G.T)), np.linalg.inv(V_sqrt))
        elif method == 'pca_cor':
            W = np.dot(np.dot(np.diag(1.0 / np.sqrt(Theta + 1e-5)), G.T), np.linalg.inv(V_sqrt))
    else:
        raise Exception('Whitening method not found.')

    return np.dot(X_centered, W.T)


def main():
    argument_parser = ArgumentParser()
    argument_parser.add_argument("--matrix", type=str, required=True)
    argument_parser.add_argument("--method", type=str, required=False, choices=['zca', 'pca', 'cholesky',
                                                                                'zca_cor', 'pca_cor'], default='zca')

    args = argument_parser.parse_args()

    matrix_path: str
    matrix_path = args.matrix
    matrix_whitened = None
    if matrix_path.endswith(".npy"):
        matrix_whitened = whiten(np.load(matrix_path), method=args.method)
    elif matrix_path.endswith(".npz"):
        matrix_whitened = whiten(np.loadz(matrix_path).toarray(), method=args.method)

    output_path = matrix_path[:-4] + f"_whitened-{args.method}.npy"
    np.save(output_path, matrix_whitened)


if __name__ == '__main__':
    main()
