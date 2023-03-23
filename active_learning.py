import os
import random

import torch


def farthest_distance_sample_dense(all_features, id2idx, sample_num, dist_func, init_ids=[], topk=None):
    """
    Args:
        all_features: 所有的特征
        id2idx: 这里应该是一个特征映射list，每个元素都是一个tensor代表这个tensor对应的特征下标，比如如果一张图片对应五个特征的话就是[[0,1,2,3,4],[5,6,7,8,9],...]
        sample_num: 需要选择的样本数量，此处指图片
        dist_func: 距离函数，partial(get_distance,type=‘cosine’)
        init_ids: 初始化第一个样本的id，可以为空
        topk:

    Returns:
        选择出的样本id
    """
    if len(init_ids) >= sample_num:
        print("Initial samples are enough")
        return init_ids

    feature_num = all_features.shape[0]
    total_num = len(id2idx)
    if total_num <= sample_num:
        print("No enough features")
        return list(range(total_num))

    idx2id = []
    for id in id2idx:
        idxs = id2idx[id]
        idx2id.extend([id] * idxs.shape[0])
    assert len(idx2id) == feature_num

    if len(init_ids) == 0:
        import random
        sample_ids = random.sample(range(total_num), 1)
    else:
        sample_ids = init_ids

    distances = torch.zeros(feature_num).cuda() + 1e20
    print(torch.max(distances, dim=0)[0])

    for i, init_id in enumerate(sample_ids):
        distances = update_distance_dense(distances, all_features, all_features[id2idx[init_id]], dist_func)
        if i % 100 == 1:
            print(i, torch.max(distances, dim=0)[0], "random")
            print(all_features.shape, all_features[id2idx[init_id]].shape)

    while len(sample_ids) < sample_num:
        new_featid = torch.max(distances, dim=0)[1]
        new_id = idx2id[new_featid]
        distances = update_distance_dense(distances, all_features, all_features[id2idx[new_id]], dist_func)
        sample_ids.append(new_id)
        if len(sample_ids) % 100 == 1:
            print(len(sample_ids))
            print(len(sample_ids), torch.max(distances, dim=0)[0], "FDS")
            print(all_features.shape, all_features[id2idx[new_id]].shape)
    assert len(set(sample_ids)) == sample_num
    return sample_ids


def get_distance(p1, p2, type, slice=1000):
    import torch.nn.functional as F
    if len(p1.shape) > 1:
        if len(p2.shape) == 1:
            # p1 (n, dim)
            # p2 (dim)
            p2 = p2.unsqueeze(0)  # (1, dim)
            if type == "cosine":
                p1 = F.normalize(p1, p=2, dim=1)
                p2 = F.normalize(p2, p=2, dim=1)
                dist = []
                iter = p1.shape[0] // slice + 1
                for i in range(iter):
                    dist_ = 1 - torch.sum(p1[slice * i:slice * (i + 1)] * p2, dim=1)  # (slice, )
                    dist.append(dist_)
                dist = torch.cat(dist, dim=0)
            elif type == "euclidean":
                dist = []
                iter = p1.shape[0] // slice + 1
                for i in range(iter):
                    dist_ = torch.norm(p1[slice * i:slice * (i + 1)] - p2, p=2, dim=1)  # (slice, )
                    dist.append(dist_)
                dist = torch.cat(dist, dim=0)
            else:
                raise NotImplementedError
        else:
            # p1 (n, dim)
            # p2 (m, dim)
            if type == "cosine":
                p1 = p1.unsqueeze(1)  # (n, 1, dim)
                p2 = p2.unsqueeze(0)  # (1, m, dim)
                p2 = F.normalize(p2, p=2, dim=2)
                dist = []
                iter = p1.shape[0] // slice + 1
                for i in range(iter):
                    p1_slice = F.normalize(p1[slice * i:slice * (i + 1)], p=2, dim=2)
                    dist_ = 1 - torch.sum(p1_slice * p2, dim=2)  # (slice, m)
                    dist.append(dist_)
                dist = torch.cat(dist, dim=0)

            elif type == "euclidean":
                p1 = p1.unsqueeze(1)  # (n, 1, dim)
                p2 = p2.unsqueeze(0)  # (1, m, dim)
                dist = []
                iter = p1.shape[0] // slice + 1
                for i in range(iter):
                    dist_ = torch.norm(p1[slice * i:slice * (i + 1)] - p2, p=2, dim=2)  # (slice, m)
                    dist.append(dist_)
                dist = torch.cat(dist, dim=0)
            else:
                raise NotImplementedError
    else:
        # p1 (dim, )
        # p2 (dim, )
        if type == "cosine":
            dist = 1 - torch.sum(p1 * p2)
        elif type == "euclidean":
            dist = torch.norm(p1 - p2, p=2)
        else:
            raise NotImplementedError
    return dist


def update_distance_dense(distances, all_features, cfeatures, dist_func):
    # all_features: (n, c)
    # cfeatures: (r, c)
    new_dist = dist_func(all_features, cfeatures)  # (n, r)
    new_dist = torch.min(new_dist, dim=1)[0]  # (n, )
    distances = torch.where(distances < new_dist, distances, new_dist)  # 合并tensor 如果满足条件取第二个参数的值，否则使用第三个参数的值
    # 这一行更新位置的意思是如果我新选择的点使得其他点的距离当前点的距离更小就使用更小的距离，否则保持原本的距离
    return distances


def write_to_file(filename, img_path_list):
    temp_list = []
    for img_path in img_path_list:
        temp_list.append(str(img_path) + '\n')
    with open(filename, "w") as file:
        file.writelines(temp_list)


def read_from_file(filename, img_count=None):
    img_path_list = []
    if os.path.exists(filename):
        with open(filename, "r") as file:
            lines = file.readlines()
            for line in lines:
                img_path_list.append(line.replace('\n', ''))
    if img_count:
        img_path_list = img_path_list[0:img_count]
    return img_path_list


def add_list_to_list(source_list, target_list):
    geal_file_list_set = set(target_list)
    for i in range(0, len(source_list)):
        if not geal_file_list_set.__contains__(source_list[i]):
            target_list.append(source_list[i])


def geal_sampling(model, sampling_loader, sample_num, geal_file_list, sample_use_al, geal_file_name, get_feature=False):
    # todo 测试流程能不能跑通
    # select_sample_file = geal_file_list
    # write_to_file(geal_file_name, select_sample_file)
    # return select_sample_file
    # 使用geal进行取样
    print("此处传入的 geal_file_list 长度为 {}".format(str(len(geal_file_list))))
    geal_file_list_set = set(geal_file_list)
    torch.cuda.empty_cache()
    model.set_mode_sampling(True)
    sampling_loader_iter = iter(sampling_loader)
    select_sample_file = []
    # 获取所有的图片名和图片中经过encoder编码后提取出来的特征,并且每个图片的特征经过kmeans聚类后提取出最重要的几个特征
    img_list_all = []
    img_features_list_all = []
    num_clusters = 5
    sample_num = min(sample_num, len(sampling_loader.dataset.dataset.dataset))
    from kmeans_pytorch import kmeans
    from functools import partial
    import math
    for i in range(0, math.ceil(len(sampling_loader.dataset.dataset.dataset) / sampling_loader.batch_size)):
        data = next(sampling_loader_iter)
        if not sample_use_al:
            # 判断是否使用geal选择样本
            img_list_all.extend([x['file_name'] for x in data])
            continue
        img_feature_list = model(data)  # bz * wh * channels
        # attn_sort, idx_sort = torch.sort(img_feature_list, dim=1, descending=False)  # 对特征attn进行排序
        # attn_cum = torch.cumsum(attn_sort, dim=1)  # (bs, wh) 对每张图片的最终的的特征图进行累加操作，进行过归一化理论上每个数组最后一个数字都是1
        # mask = attn_cum > 0.5  # 其中当累加值大于0.5为true,把学习率之下的特征全部删除掉
        for b in range(img_feature_list.shape[0]):
            # mask[b][idx_sort[b]] = mask[b].clone()
            # if torch.sum(mask[b]) > 0:
            #     img_feature_list[b] = img_feature_list[b][mask[b]]  # 保留学习率筛选下的特征
            # todo 此处需要修改kmeans源码因为初始化的种子可能会出现问题导致死循环
            cluster_ids_x, cluster_centers = kmeans(X=img_feature_list[b], num_clusters=num_clusters,
                                                    distance='euclidean',
                                                    device=torch.device('cuda:0'))
            # count += cluster_centers.shape[0]
            img_features_list_all.append(cluster_centers.cuda())  # cluster_centers: num_clusters * channels
            # img_features_list_all.extend(img_feature_list[b].cpu())
        img_list_all.extend([x['file_name'] for x in data])
    if get_feature:
        import numpy as np
        temp_feature_list = np.asarray(
            [temp_feature.cpu().numpy() for temp_feature in torch.cat(img_features_list_all, dim=0)])
        np.savetxt('temp_feature_list.csv', temp_feature_list, delimiter=',', fmt='%f', newline='\n')
        print('这个时候应该把所有的特征全部保存为文件')
        return
    if sample_use_al:
        # img_features_list_all = torch.cat(img_features_list_all, dim=0)
        # 对已经提取出来的特征进行挑选
        torch.cuda.empty_cache()
        img_features_list_all = torch.cat(img_features_list_all,
                                          dim=0).cuda()  # image * num_clusters * channels -> (image * num_clusters) * channels
        id2idx = {}
        init_ids = []
        for i in range(0, len(img_list_all)):
            id2idx[i] = torch.arange(i * num_clusters, (i + 1) * num_clusters)
            if geal_file_list_set.__contains__(img_list_all[i]):
                init_ids.append(i)
        print("此处初始化的 init_ids 长度为 {}".format(str(len(init_ids))))
        select_samples = farthest_distance_sample_dense(img_features_list_all, id2idx, sample_num,
                                                        partial(get_distance, type='cosine'), init_ids=init_ids,
                                                        topk=None)
        for i in select_samples:
            select_sample_file.append(img_list_all[i])
    else:
        # 不使用al则直接随机挑选
        select_sample_file_temp = random.sample(img_list_all, sample_num)
        for img_path in select_sample_file_temp:
            if geal_file_list_set.__contains__(img_path):
                select_sample_file_temp.remove(img_path)
        if len(geal_file_list) + len(select_sample_file_temp) > sample_num:
            select_sample_file_temp = select_sample_file_temp[0:sample_num - len(geal_file_list)]
        select_sample_file.extend(geal_file_list)
        select_sample_file.extend(select_sample_file_temp)
    write_to_file(geal_file_name, select_sample_file)
    return select_sample_file
