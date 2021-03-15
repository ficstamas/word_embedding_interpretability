import numpy as np
from src.modules.utilities.labels import Labels
from src.modules.utilities.logging import Logger
import os
import json


def test(embedding: np.ndarray, labels: Labels, **kwargs):
    log = Logger().logger

    log.info("Calculating Accuracy...")
    etalon = []
    dataset = []
    unique_datasets = {}
    labeled_filter = []
    for i, l in enumerate(labels.labels):
        if len(l) > 0:
            etalon.append(l)
            dataset.append(labels.dataset[i])
            unique_datasets[labels.dataset[i]] = []
            labeled_filter.append(i)

    unique_datasets["ALL"] = []

    prediction: np.ndarray
    prediction = np.argmax(embedding, axis=1)[np.array(labeled_filter)]
    log.debug(f"Prediction Object: type {type(prediction)};")
    log.debug(f"Prediction Object: shape {prediction.shape};")

    for i in range(prediction.shape[0]):
        if labels.i2l[prediction[i]] in etalon[i]:
            unique_datasets[dataset[i]].append(1)
            unique_datasets["ALL"].append(1)
        else:
            unique_datasets[dataset[i]].append(0)
            unique_datasets["ALL"].append(0)
    log.info("Results: ")
    results = {}
    for key in unique_datasets:
        results[key] = np.mean(unique_datasets[key])
        log.info(f"{key}: {results[key]}")

    if kwargs['save']:
        out_path = os.path.join(kwargs["path"], "results/argmax.json")
        with open(out_path, mode="w") as f:
            json.dump(results, f, indent=4)
        log.info(f"Results are dumped at: {out_path}")
    return results
