import os
import lmdb
from joblib import Parallel, delayed, cpu_count
from tqdm import tqdm


def remove_labels_from_lmdb_dataset(input_lmdb_path, output_lmdb_path):
    """
    Create LMDB dataset for training and evaluation.
    ARGS:
        input_lmdb_path  : input folder path where starts imagePath
        output_lmdb_path : LMDB output path
    """
    os.makedirs(output_lmdb_path, exist_ok=True)
    cache = {}
    cnt = 1
    env_input = lmdb.open(input_lmdb_path, max_readers=32, readonly=True, lock=False, readahead=False, meminit=False)
    env_output = lmdb.open(output_lmdb_path, map_size=1099511627776)
    with env_input.begin(write=False) as txn:
        n_samples = int(txn.get('num-samples'.encode()))
        for _ in tqdm(range(n_samples)):
            image_key_code = 'image-%09d'.encode() % cnt
            image_key = txn.get(image_key_code)
            cache[image_key_code] = image_key

            label_key_code = 'label-%09d'.encode() % cnt
            cache[label_key_code] = 'unlabeleddata'.encode()

            if cnt % 1000 == 0:
                write_cache(env_output, cache)
                cache = {}
            cnt += 1
        cache['num-samples'.encode()] = str(n_samples).encode()
        write_cache(env_output, cache)


def write_cache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            txn.put(k, v)


if __name__ == '__main__':
    labeled_data_root = "data/training/label/real"
    unlabeled_data_root = "data/training/label_without_labels/real"
    datasets = [
        "10.MLT19",
        "11.ReCTS",
        "1.SVT",
        "2.IIIT",
        "3.IC13",
        "4.IC15",
        "5.COCO",
        "6.RCTW17",
        "7.Uber",
        "8.ArT",
        "9.LSVT",
    ]

    n_jobs = min(cpu_count(), len(datasets))
    Parallel(n_jobs=n_jobs)(delayed(remove_labels_from_lmdb_dataset)(
        input_lmdb_path=os.path.join(labeled_data_root, dataset),
        output_lmdb_path=os.path.join(unlabeled_data_root, dataset)) for dataset in datasets)
