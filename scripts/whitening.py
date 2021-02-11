import numpy as np
from argparse import ArgumentParser
import os
from scipy import sparse as sp

import logging
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')


def whiten(X: np.ndarray, Y=None, method='zca'):
    """
    Source: https://gist.github.com/joelouismarino/ce239b5601fff2698895f48003f7464b
    Whitens the input matrix X using specified whitening method.
    Inputs:
        X:      Input data matrix with data examples along the first dimension
        method: Whitening method. Must be one of 'zca', 'zca_cor', 'pca',
                'pca_cor', or 'cholesky'.
    """
    X = X.reshape((-1, np.prod(X.shape[1:])))
    Y = Y.reshape((-1, np.prod(Y.shape[1:])))
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
        P = np.dot(np.dot(np.linalg.pinv(V_sqrt), Sigma), np.linalg.pinv(V_sqrt))
        G, Theta, _ = np.linalg.svd(P)
        if method == 'zca_cor':
            W = np.dot(np.dot(G, np.dot(np.diag(1.0 / np.sqrt(Theta + 1e-5)), G.T)), np.linalg.pinv(V_sqrt))
        elif method == 'pca_cor':
            W = np.dot(np.dot(np.diag(1.0 / np.sqrt(Theta + 1e-5)), G.T), np.linalg.pinv(V_sqrt))
    else:
        raise Exception('Whitening method not found.')

    if Y is None:
        return np.dot(X_centered, W.T), None
    else:
        return np.dot(X_centered, W.T), np.dot(Y - np.mean(Y, axis=0), W.T)


def main():
    argument_parser = ArgumentParser()
    argument_parser.add_argument("--matrix", type=str, required=True)
    argument_parser.add_argument("--test", type=str, required=False, default=None)
    argument_parser.add_argument("--method", type=str, required=False, choices=['zca', 'pca', 'cholesky',
                                                                                'zca_cor', 'pca_cor'], default='zca')
    argument_parser.add_argument("--output_folder", type=str, required=False, default=None)

    args = argument_parser.parse_args()

    logging.info(f"Whitening ({args.method}) file {args.matrix}")

    matrix_path: str
    matrix_path = args.matrix
    matrix_whitened = None
    test_whitened = None
    try:
        if matrix_path.endswith(".npy"):
            matrix_whitened, test_whitened = whiten(np.load(matrix_path), Y=np.load(args.test), method=args.method)
        elif matrix_path.endswith(".npz"):
            matrix_whitened, test_whitened = whiten(sp.load_npz(matrix_path).toarray(), Y=sp.load_npz(args.test).toarray(),
                                                    method=args.method)
    except ValueError as ve:
        logging.error(str(ve))
        logging.error(f"Error occurred during the whitening of file: {args.method}")

    if args.output_folder is None:
        output_path = matrix_path[:-4] + f"_whitened-{args.method}.npy"
        np.save(output_path, matrix_whitened)
        logging.info(f"Done, saved to {output_path}\n")
    else:
        filename = os.path.basename(matrix_path)[:-4] + f"_whitened-{args.method}.npy"
        output_path = os.path.join(args.output_folder, filename)
        np.save(output_path, matrix_whitened)

    if args.test is None:
        output_path = args.test[:-4] + f"_whitened-{args.method}.npy"
        np.save(output_path, test_whitened)
        logging.info(f"Test, saved to {output_path}\n")
    else:
        filename = os.path.basename(args.test)[:-4] + f"_whitened-{args.method}.npy"
        output_path = os.path.join(args.output_folder, filename)
        np.save(output_path, test_whitened)
        logging.info(f"Test, saved to {output_path}\n")


if __name__ == '__main__':
    main()
