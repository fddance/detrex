import itertools
import logging
import random

import pandas as pd
import torch
from kmeans_pytorch import kmeans
from tabulate import tabulate
from termcolor import colored

from detectron2.config import LazyConfig, instantiate
from detectron2.engine import (
    SimpleTrainer,
    default_argument_parser,
    default_setup,
    default_writers,
    hooks,
    launch,
)
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from detectron2.utils.logger import log_first_n


def Input():
    # 从文件中读取特征
    sample = pd.read_csv('temp_feature_list.csv', header=None)
    sample = Pre_Data(sample)
    return sample


def Pre_Data(data):
    # 预处理特征
    [N, L] = np.shape(data)
    scaler = MinMaxScaler()
    scaler.fit(data)
    NewData = scaler.transform(data)
    return NewData


def print_instances_class_histogram(dataset_dicts, class_names, hist_togram_all=None):
    """
    Args:
        dataset_dicts (list[dict]): list of dataset dicts.
        class_names (list[str]): list of class names (zero-indexed).
    """
    num_classes = len(class_names)
    histogram = get_class_instance_count(dataset_dicts, class_names)

    N_COLS = min(8, len(class_names) * 2)

    def short_name(x):
        # make long class names shorter. useful for lvis
        if len(x) > 13:
            return x[:11] + ".."
        return x

    if hist_togram_all is not None:
        data = list(
            itertools.chain(
                *[[short_name(class_names[i]), int(v), int(hist_togram_all[i]),
                   "{:.3f}".format(float(int(v) / int(hist_togram_all[i])))]
                  for i, v in enumerate(histogram)])
        )
        total_num_instances = sum(histogram)
        data.extend([None] * (N_COLS - (len(data) % N_COLS)))
        if num_classes > 1:
            data.extend(["total", total_num_instances])
        data = itertools.zip_longest(*[data[i::N_COLS] for i in range(N_COLS)])
        table = tabulate(
            data,
            headers=["category", "#instance", "#all_ins", "scale"] * (N_COLS // 4),
            tablefmt="pipe",
            numalign="left",
            stralign="center",
        )
    else:
        data = list(
            itertools.chain(*[[short_name(class_names[i]), int(v)] for i, v in enumerate(histogram)])
        )
        total_num_instances = sum(data[1::2])
        data.extend([None] * (N_COLS - (len(data) % N_COLS)))
        if num_classes > 1:
            data.extend(["total", total_num_instances])
        data = itertools.zip_longest(*[data[i::N_COLS] for i in range(N_COLS)])
        table = tabulate(
            data,
            headers=["category", "#instances"] * (N_COLS // 2),
            tablefmt="pipe",
            numalign="left",
            stralign="center",
        )
    print('\n' + table)


def get_class_instance_count(dataset_dicts, class_names):
    # 获取每个class的实例数量
    num_classes = len(class_names)
    hist_bins = np.arange(num_classes + 1)
    histogram = np.zeros((num_classes,), dtype=np.int)
    for entry in dataset_dicts:
        annos = entry["annotations"]
        classes = np.asarray(
            [x["category_id"] for x in annos if not x.get("iscrowd", 0)], dtype=np.int
        )
        if len(classes):
            assert classes.min() >= 0, f"Got an invalid category_id={classes.min()}"
            assert (
                    classes.max() < num_classes
            ), f"Got an invalid category_id={classes.max()} for a dataset of {num_classes} classes"
        histogram += np.histogram(classes, bins=hist_bins)[0]
    return histogram


class_names = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
               'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']


def filter_dataset_sample(data_list):
    sample_id_list = load_sample_id()
    while 5717 in sample_id_list:
        sample_id_list.remove(5717)
    sub_sample_size = len(sample_id_list) / len(data_list)
    print("选出数据占总数据的 {:.3f}".format(sub_sample_size))
    data_result = [data_list[i] for i in sample_id_list]
    print_instances_class_histogram(data_result, class_names,
                                    hist_togram_all=get_class_instance_count(data_list, class_names))

    sample_id_list = random.sample([i for i in range(len(data_list))], int(sub_sample_size * len(data_list)))
    data_result = [data_list[i] for i in sample_id_list]
    print_instances_class_histogram(data_result, class_names,
                                    hist_togram_all=get_class_instance_count(data_list, class_names))
    return data_result


def load_sample_id():
    cluster_ids, cluster_centers = kmeans(X=torch.from_numpy(Input()), num_clusters=20, distance='euclidean',
                                          device=torch.device('cuda:0'))
    result = [[] for i in range(20)]
    sample_percent = 0.1
    sample_features = []
    sample_ids_set = set()
    sample_ids = []
    # 遍历第一个数组，将值存入对应下标的位置
    for i in range(len(cluster_ids)):
        index = cluster_ids[i]
        result[index].append(i)
    for sub_arr in result:
        sub_sample_size = int(len(sub_arr) * sample_percent)
        sub_sampled_indices = random.sample(sub_arr, sub_sample_size)
        sample_features.extend(sub_sampled_indices)
    for feature_id in sample_features:
        img_id = int(feature_id / 5)
        sample_ids_set.add(img_id)
    sample_ids = list(sample_ids_set)
    return sample_ids


if __name__ == '__main__':
    # 此处设置要加载的数据集
    from detectron2.data import DatasetCatalog
    from detectron2.data.datasets import load_coco_json

    DatasetCatalog.register('coco_2017_train_fddance', lambda: load_coco_json(
        '/mnt/84BA4F3DBA4F2ACE/ubuntu/data/dataset/coco/annotations/voc2012_dataset_train.json',
        '',
        'coco_2017_train_fddance'))
    args = default_argument_parser(config_file='projects/dino/configs/dino_r50_4scale_12ep.py', resume=False)
    args = args.parse_args()
    cfg = LazyConfig.load(args.config_file)
    cfg = LazyConfig.apply_overrides(cfg, args.opts)
    default_setup(cfg, args)
    # 这里进行加载
    train_loader_all = instantiate(cfg.dataloader.train_sample)
