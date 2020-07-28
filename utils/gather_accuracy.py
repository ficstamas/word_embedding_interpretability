import os
import json
import sys
from argparse import ArgumentParser
import random
import string


def get_random_string(length):
    letters = string.ascii_lowercase + string.ascii_uppercase
    numbers = string.digits
    result_str = ''.join(random.choice(letters + numbers) for _ in range(length))
    return result_str


def gather(workspace):
    results = {}
    if not os.path.exists(workspace):
        print(f"Path {workspace} does not exists")
        sys.exit(-1)

    for root, dirs, files in os.walk(workspace):
        if "params.config" not in files:
            continue
        # reading params
        fp = open(os.path.join(root, "params.config"))
        params = json.load(fp)
        fp.close()

        # reading accuracies
        accuracy_path = os.path.join(params['project']['results'], "accuracy.txt")
        fp = open(accuracy_path)
        res = fp.readlines()
        fp.close()

        results[get_random_string(8)] = {'accuracy_path': accuracy_path,
                                         'workspace': workspace,
                                         'name': params['project']['name'],
                                         'scores': res}

    # save
    fp = open(os.path.join(workspace, "gathered_accuracy.json"), mode='w', encoding='utf8')
    json.dump(results, fp, indent=4)
    fp.close()


if __name__ == '__main__':
    parser = ArgumentParser(description='Gather results in workspace')

    # Embedding
    parser.add_argument("--workspace", type=str, required=True,
                        help="Path to the workspace")

    args = parser.parse_args()
    gather(args.workspace)
