import matplotlib.pyplot as plt
import numpy as np
from interpretability.reader.semcat import semcat_reader_nc
from argparse import ArgumentParser


def plot_semantic_decomposition(ax, semcat: str, input_file: str, embedding_path:str, word="rövid"):
    """
    Creates a plot of semantic decomposition
    Parameters
    ----------
    semcat: str
        Path to SemCat categories directory
    input_file: str
        Path to input file
    output_file: str
        Optional: Path to save figure as PNG
    Returns
    -------
    """
    row_index = -1
    y_axis_label = []
    in_cat = []
    words = []

    semcat = semcat_reader_nc(semcat, None)

    with open(input_file, mode="r", encoding="utf8") as f:
        for i, line in enumerate(f.readlines()):
            if i == 50000:
                break
            w = line.split(' ')[0]
            if w == word:
                row_index = i
                break

    embedding = np.load(embedding_path)[row_index]
    sorted_indexes = np.argsort(embedding)[-20:]
    sorted_values = embedding[sorted_indexes]
    in_category = np.zeros(sorted_indexes.shape)
    for i in range(sorted_indexes.shape[0]):
        y_axis_label.append(semcat.i2c[sorted_indexes[i]])
        if word in semcat.vocab[semcat.i2c[sorted_indexes[i]]]:
            in_category[i] = 1

    y_pos = np.arange(len(y_axis_label))

    barlist = ax.barh(y_pos, sorted_values, align='center')

    for i in range(in_category.shape[0]):
        if in_category[i] == 1:
            barlist[i].set_color('r')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(y_axis_label)
    ax.invert_yaxis()
    ax.set_xlabel('')


def main():
    plt.rcdefaults()
    fig, axs = plt.subplots(2, 3)
    fig.set_size_inches(15, 7)
    word = "ember"

    plot_semantic_decomposition(axs[1][0],
                                "../data/semcat/semcat_en-de-hu.json",
                                "../data/mszny/wv/wiki.hu.vec",
                                "../data/mszny/mszny50000/wiki.hu.align.vec-hellinger_normal-K2000-l0.05_sparse_interpret_0.0/saves/transformed_space.npy",
                                word)

    plot_semantic_decomposition(axs[1][1],
                                "../data/semcat/semcat_en-de-hu.json",
                                "../data/mszny/wv/wiki.hu.align.vec",
                                "../data/mszny/mszny50000/wiki.hu.align.vec-hellinger_normal-K2000-l0.05_sparse_interpret_0.0/saves/transformed_space.npy",
                                word)

    plot_semantic_decomposition(axs[1][2],
                                "../data/semcat/semcat_en-de-hu.json",
                                "../data/mszny/wv/hu.szte.w2v.fasttext.vec",
                                "../data/mszny/mszny50000/hu.szte.w2v.fasttext.vec-hellinger_normal-K2000-l0.05_sparse_interpret_0.0/saves/transformed_space.npy",
                                word)

    plot_semantic_decomposition(axs[0][0],
                                "../data/semcat/semcat_en-de-hu.json",
                                "../data/mszny/wv/wiki.hu.vec",
                                "../data/mszny/mszny50000/wiki.hu.vec-hellinger_interpret_0.0/saves/transformed_space.npy",
                                word)

    plot_semantic_decomposition(axs[0][1],
                                "../data/semcat/semcat_en-de-hu.json",
                                "../data/mszny/wv/wiki.hu.align.vec",
                                "../data/mszny/mszny50000/wiki.hu.align.vec-hellinger_interpret_0.0/saves/transformed_space.npy",
                                word)

    plot_semantic_decomposition(axs[0][2],
                                "../data/semcat/semcat_en-de-hu.json",
                                "../data/mszny/wv/hu.szte.w2v.fasttext.vec",
                                "../data/mszny/mszny50000/hu.szte.w2v.fasttext.vec-hellinger_interpret_0.0/saves/transformed_space.npy",
                                word)

    fig.suptitle(f'Semantic decomposition of word "{word}"')
    axs[0][0].set_title("Fasttext HU")
    axs[0][1].set_title("Fasttext Aligned")
    axs[0][2].set_title("Szeged")
    # fig.tight_layout() bbox_inches="tight" pad=0.4, w_pad=0.5, h_pad=1.0
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("semantic_decomposition_ember.pdf", )


if __name__ == '__main__':
    main()