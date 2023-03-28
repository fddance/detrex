import itertools
import logging
import math
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
import active_learning
from sklearn.preprocessing import MinMaxScaler
from detectron2.utils.logger import log_first_n
import csv


def Input():
    # 从文件中读取特征
    sample = pd.read_csv('temp_feature_list_coco2017.csv', header=None)
    sample = Pre_Data(sample)
    return sample


def Pre_Data(data):
    # 预处理特征
    [N, L] = np.shape(data)
    scaler = MinMaxScaler()
    scaler.fit(data)
    NewData = scaler.transform(data)
    return NewData


def print_instances_class_histogram(dataset_dicts, class_names, sub_sample_size, hist_togram_all=None):
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
        class_ratio = [float(int(v) / int(hist_togram_all[i])) for i, v in enumerate(histogram)]
        print("当前的方差为 {} ".format(sum([(x - sub_sample_size) ** 2 for x in class_ratio]) / len(class_ratio)))
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
    print(table + '\n')


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


# class_names = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
#                'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
class_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
               'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
               'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
               'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
               'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
               'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
               'scissors', 'teddy bear', 'hair drier', 'toothbrush']
class_count = len(class_names)
sample_percent = 0.05
al_file = 'voc_al_file_ids.txt'
random_file = 'voc_random_file_ids.txt'
sample_id_list_file_name = 'sample_id_list.txt'


def read_from_file_and_print(data_list):
    # 查看当前选择出的样本占总体的比例
    geal_file_list = active_learning.read_from_file('bak_geal_file_list_random.txt')
    # geal_file_list = active_learning.read_from_file('geal_file_list_random.txt')
    patch_count = 5863
    data_result = []
    for i in range(0, int(np.ceil(len(geal_file_list) / patch_count))):
        geal_file_list_set = set(geal_file_list[i * patch_count:min((i + 1) * patch_count, len(geal_file_list))])
        for data in data_list:
            if geal_file_list_set.__contains__(data['file_name']):
                data_result.append(data)
        sub_sample_size = len(data_result) / len(data_list)
        print("选出数据占总数据的 {:.3f}".format(sub_sample_size))
        print_instances_class_histogram(data_result, class_names, sub_sample_size,
                                        hist_togram_all=get_class_instance_count(data_list, class_names))


def filter_dataset_sample(data_list):
    read_from_file_and_print(data_list)
    return data_list
    sample_id_list = active_learning.read_from_file(sample_id_list_file_name)
    sub_sample_size = len(sample_id_list) / len(data_list)
    print("选出数据占总数据的 {:.3f}".format(sub_sample_size))
    data_result = [data_list[int(i)] for i in sample_id_list]
    active_learning.write_to_file(al_file, [data['file_name'] for data in data_result])
    print_instances_class_histogram(data_result, class_names, sub_sample_size,
                                    hist_togram_all=get_class_instance_count(data_list, class_names))

    sample_id_list = random.sample([i for i in range(len(data_list))], len(sample_id_list))
    data_result = [data_list[i] for i in sample_id_list]
    active_learning.write_to_file(random_file, [data['file_name'] for data in data_result])
    print_instances_class_histogram(data_result, class_names, sub_sample_size,
                                    hist_togram_all=get_class_instance_count(data_list, class_names))
    return data_result


def load_sample_id():
    feature_list = Input()
    patch_count = 50000
    sample_ids_all = []
    label_result = [[] for i in range(0, int(np.ceil(len(feature_list) / patch_count)))]
    # label_result = []
    # with open('labels_result.csv', 'r') as f:
    #     reader = csv.reader(f)
    #     for row in reader:
    #         label_result.append([int(x) for x in row])
    for i in range(0, int(np.ceil(len(feature_list) / patch_count))):
        temp_feature_list = feature_list[i * patch_count:min((i + 1) * patch_count, len(feature_list))]
        cluster_ids, cluster_centers = kmeans(X=torch.from_numpy(temp_feature_list), num_clusters=class_count,
                                              distance='euclidean',
                                              device=torch.device('cuda:0'))
        label_result[i].extend(cluster_ids.numpy())
        continue  # todo 先将特征数组保存下来
        cluster_ids = label_result[i]
        result = [[] for i in range(class_count)]
        sample_feature_result = [[] for i in range(class_count)]
        sample_img_result = [[] for i in range(class_count)]
        sample_img_count = 0
        sample_features = []
        sample_ids_set = set()
        last_cluster_sample_ids = []
        sample_result_count = round(len(cluster_ids) / 5 * sample_percent)
        # 遍历第一个数组，将值存入对应下标的位置
        for j in range(len(cluster_ids)):
            index = cluster_ids[j]
            result[index].append(j)
        for j in range(len(result)):
            sub_arr = result[j]
            sub_sample_size = int(len(sub_arr) * sample_percent)
            sub_sampled_indices = random.sample(sub_arr, sub_sample_size)  # 每个集群的数组中应该取出差不多数量的特征
            sample_feature_result[j].extend(sub_sampled_indices)
            sample_img_result[j].extend(
                [int((i * patch_count + feature_id) / 5) for feature_id in sub_sampled_indices])  # 将特征转换为图片
            temp_sample_count = np.ceil(len(sub_sampled_indices) / 5)
            count = 0
            for img_id in sample_img_result[j]:  # 每个特征集群中仅添加当前特征集群大小比例的图片数量
                if img_id not in sample_ids_set:
                    sample_ids_set.add(img_id)
                    count = count + 1
                    if count == temp_sample_count:  # 最后一个放入一个临时数组中，如果最后数量超过则从中抽样删除
                        last_cluster_sample_ids.append(img_id)
                        break
            sample_img_count += temp_sample_count
            sample_features.extend(sub_sampled_indices)
        sample_ids = list(sample_ids_set)
        if sample_result_count != sample_img_count:
            print("结果数量对不上，应该选出 {} ，实际上选出了 {} ".format(sample_result_count, sample_img_count))
            remove_sample_ids = random.sample(last_cluster_sample_ids, int(sample_img_count - sample_result_count))
            sample_ids = [i for i in sample_ids if i not in remove_sample_ids]
        sample_ids_all.extend(sample_ids)
    # 将三维数组写入文件
    import csv
    with open('labels_result.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        for row in label_result:
            writer.writerow(row)
    raise Exception('现在已经全部将label写出')
    active_learning.write_to_file(sample_id_list_file_name, sample_ids_all)
    return sample_ids_all


if __name__ == '__main__':
    # 此处设置要加载的数据集
    from detectron2.data import DatasetCatalog
    from detectron2.data.datasets import load_coco_json

    load_sample_id()
    # DatasetCatalog.register('coco_2017_train_fddance', lambda: load_coco_json(
    #     # '/mnt/84BA4F3DBA4F2ACE/ubuntu/data/dataset/coco/annotations/voc2012_dataset_train.json',
    #     '/mnt/84BA4F3DBA4F2ACE/ubuntu/data/dataset/coco/annotations/instances_train2017.json',
    #     '/mnt/84BA4F3DBA4F2ACE/ubuntu/data/dataset/coco/train2017',
    #     'coco_2017_train_fddance'))
    DatasetCatalog.register('coco_2017_train_fddance', lambda: load_coco_json(
        '/mnt/84BA4F3DBA4F2ACE/ubuntu/data/dataset/coco/annotations/instances_train2017.json',
        '/mnt/84BA4F3DBA4F2ACE/ubuntu/data/dataset/coco/train2017',
        'coco_2017_train_fddance'))
    args = default_argument_parser(config_file='projects/dino/configs/dino_r50_4scale_12ep.py', resume=False)
    args = args.parse_args()
    cfg = LazyConfig.load(args.config_file)
    cfg = LazyConfig.apply_overrides(cfg, args.opts)
    default_setup(cfg, args)
    # 这里进行加载
    train_loader_all = instantiate(cfg.dataloader.train_sample)
