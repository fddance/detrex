#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
"""
Training script using the new "LazyConfig" python config files.

This scripts reads a given python config file and runs the training or evaluation.
It can be used to train any models or dataset as long as they can be
instantiated by the recursive construction defined in the given config file.

Besides lazy construction of models, dataloader, etc., this scripts expects a
few common configuration parameters currently defined in "configs/common/train.py".
To add more complicated training logic, you can easily add other configs
in the config file and implement a new train_net.py to handle them.
"""
import logging
import os
import sys
import time
import torch
from torch.nn.parallel import DataParallel, DistributedDataParallel

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import LazyConfig, instantiate

from detectron2.engine import (
    SimpleTrainer,
    default_argument_parser,
    default_setup,
    default_writers,
    hooks,
    launch,
)
from detectron2.engine.defaults import create_ddp_model
from detectron2.evaluation import inference_on_dataset, print_csv_format
from detectron2.utils import comm
import active_learning
from load_image_test import print_instances_class_histogram, get_class_instance_count

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

logger = logging.getLogger("detrex")

activate_learning_flag = False  # todo 如果使用主动学习打开该选项
sample_use_al = False   # 挑选样本是否使用al，false时为随机
train_resume = True    # 是否恢复
geal_file_name = 'geal_file_list_al.txt' if sample_use_al else 'geal_file_list_random.txt'
base_sample_count = 5863     # 基础选择数量，voc总样本数5717，此处取10%
epoch = 5  # 主动学习每次选择样本后迭代多少次

# 打印coco数据集信息
class_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
               'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
               'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
               'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
               'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
               'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
               'scissors', 'teddy bear', 'hair drier', 'toothbrush']

class Trainer(SimpleTrainer):
    """
    We've combine Simple and AMP Trainer together.
    """

    def __init__(
        self,
        model,
        dataloader,
        optimizer,
        amp=False,
        clip_grad_params=None,
        grad_scaler=None,
    ):
        super().__init__(model=model, data_loader=dataloader, optimizer=optimizer)
        self.temp_max_iter = 9999
        unsupported = "AMPTrainer does not support single-process multi-device training!"
        if isinstance(model, DistributedDataParallel):
            assert not (model.device_ids and len(model.device_ids) > 1), unsupported
        assert not isinstance(model, DataParallel), unsupported

        if amp:
            if grad_scaler is None:
                from torch.cuda.amp import GradScaler

                grad_scaler = GradScaler()
            self.grad_scaler = grad_scaler

        # set True to use amp training
        self.amp = amp

        # gradient clip hyper-params
        self.clip_grad_params = clip_grad_params

    def run_step(self):
        """
        Implement the standard training logic described above.
        """
        assert self.model.training, "[Trainer] model was changed to eval mode!"
        assert torch.cuda.is_available(), "[Trainer] CUDA is required for AMP training!"
        from torch.cuda.amp import autocast

        start = time.perf_counter()
        """
        If you want to do something with the data, you can wrap the dataloader.
        """
        data = next(self._data_loader_iter)
        data_time = time.perf_counter() - start

        """
        If you want to do something with the losses, you can wrap the model.
        """
        loss_dict = self.model(data)
        with autocast(enabled=self.amp):
            if isinstance(loss_dict, torch.Tensor):
                losses = loss_dict
                loss_dict = {"total_loss": loss_dict}
            else:
                losses = sum(loss_dict.values())

        """
        If you need to accumulate gradients or do something similar, you can
        wrap the optimizer with your custom `zero_grad()` method.
        """
        self.optimizer.zero_grad()

        if self.amp:
            self.grad_scaler.scale(losses).backward()
            if self.clip_grad_params is not None:
                self.grad_scaler.unscale_(self.optimizer)
                self.clip_grads(self.model.parameters())
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            losses.backward()
            if self.clip_grad_params is not None:
                self.clip_grads(self.model.parameters())
            self.optimizer.step()

        self._write_metrics(loss_dict, data_time)

    def clip_grads(self, params):
        params = list(filter(lambda p: p.requires_grad and p.grad is not None, params))
        if len(params) > 0:
            return torch.nn.utils.clip_grad_norm_(
                parameters=params,
                **self.clip_grad_params,
            )

    def reset_data_loader_local(self, data_loader_new):
        del self.data_loader
        data_loader = data_loader_new
        self.data_loader = data_loader
        self._data_loader_iter_obj = None


def freeze_backbone(model):
    logger.info("开始冻结网络")
    # 冻结所有特征提取网络,其他地方直接开始训练
    for k, v in model.named_parameters():
        if 'backbone' in k:
            v.requires_grad = False


def filter_dataset_dicts(data_list):
    if not activate_learning_flag:
    # if True:
        return data_list
    logger.info("此处进行样本的筛选")
    geal_file_list = active_learning.read_from_file(geal_file_name)
    data_result = []
    if len(geal_file_list) < 1:
        # 第一次随机初始化
        import random
        logger.info('现在是第一次加载，随机选取 {} 个样本'.format(base_sample_count))
        list_index = random.sample(range(0, len(data_list)), base_sample_count)
        for index in list_index:
            data_result.append(data_list[index])
            geal_file_list.append(data_list[index]['file_name'])
        active_learning.write_to_file(geal_file_name, geal_file_list)
    else:
        # 之后使用每次选择出来的数字
        logger.info('现在开始主动学习选择的样本')
        geal_file_list_set = set(geal_file_list)
        for data in data_list:
            if geal_file_list_set.__contains__(data['file_name']):
                data_result.append(data)
    logger.info('本次加载了 {} 个样本进行训练'.format(str(len(data_result))))
    sub_sample_size = len(data_result) / len(data_list)
    print("选出数据占总数据的 {:.3f}".format(sub_sample_size))
    print_instances_class_histogram(data_result, class_names, sub_sample_size,
                                    hist_togram_all=get_class_instance_count(data_list, class_names))
    return data_result


def do_test(cfg, model):
    if "evaluator" in cfg.dataloader:
        ret = inference_on_dataset(
            model, instantiate(cfg.dataloader.test), instantiate(cfg.dataloader.evaluator)
        )
        print_csv_format(ret)
        return ret


def do_train(args, cfg):
    """
    Args:
        cfg: an object with the following attributes:
            model: instantiate to a module
            dataloader.{train,test}: instantiate to dataloaders
            dataloader.evaluator: instantiate to evaluator for test set
            optimizer: instantaite to an optimizer
            lr_multiplier: instantiate to a fvcore scheduler
            train: other misc config defined in `configs/common/train.py`, including:
                output_dir (str)
                init_checkpoint (str)
                amp.enabled (bool)
                max_iter (int)
                eval_period, log_period (int)
                device (str)
                checkpointer (dict)
                ddp (dict)
    """
    # 此处定义模型,模型所有相关均在此处
    model = instantiate(cfg.model)
    logger = logging.getLogger("detectron2")
    logger.info("Model:\n{}".format(model))
    model.to(cfg.train.device)

    cfg.optimizer.params.model = model
    optim = instantiate(cfg.optimizer)

    train_loader = instantiate(cfg.dataloader.train)
    # 多卡情况下,此处无用
    model = create_ddp_model(model, **cfg.train.ddp)

    trainer = Trainer(
        model=model,
        dataloader=train_loader,
        optimizer=optim,
        amp=cfg.train.amp.enabled,
        clip_grad_params=cfg.train.clip_grad.params if cfg.train.clip_grad.enabled else None,
    )

    checkpointer = DetectionCheckpointer(
        model,
        cfg.train.output_dir,
        trainer=trainer,
    )

    trainer.register_hooks(
        [
            hooks.IterationTimer(),
            hooks.LRScheduler(scheduler=instantiate(cfg.lr_multiplier)),
            hooks.PeriodicCheckpointer(checkpointer, **cfg.train.checkpointer)
            if comm.is_main_process()
            else None,
            hooks.EvalHook(cfg.train.eval_period, lambda: do_test(cfg, model)),
            hooks.PeriodicWriter(
                default_writers(cfg.train.output_dir, cfg.train.max_iter),
                period=cfg.train.log_period,
            )
            if comm.is_main_process()
            else None,
        ]
    )

    checkpointer.resume_or_load(cfg.train.init_checkpoint, resume=args.resume)
    freeze_backbone(model)  # 读取后冻结网络
    if args.resume and checkpointer.has_checkpoint():
        # The checkpoint stores the training iteration that just finished, thus we start
        # at the next iteration
        start_iter = trainer.iter + 1
    else:
        start_iter = 0
    if not activate_learning_flag:
        # print(do_test(cfg, model))
        trainer.temp_max_iter = cfg.train.max_iter
        trainer.train(start_iter, cfg.train.max_iter)
        return

    sample_count = base_sample_count
    train_loader_all = instantiate(cfg.dataloader.train_all)
    if args.resume:
        geal_file_list = active_learning.read_from_file(geal_file_name)
        sample_count = len(geal_file_list) if len(geal_file_list) > 0 else sample_count
    first_iter = True
    pass_train = False  # 这个为true表明当前阶段的训练已经完成，在挑选样本时出现问题
    while start_iter < cfg.train.max_iter:
        # active_learning.geal_sampling(trainer.model, train_loader_all, sample_count, geal_file_list, sample_use_al, geal_file_name, get_feature=True)
        # raise Exception
        logger.info('现在使用了 {} 个样本训练 {} 个epoch'.format(str(sample_count), str(epoch)))
        # 此处正常开始本次选择样本后的训练
        # 此处考虑到如果是恢复训练可能迭代不准确
        if first_iter and args.resume and sample_count > base_sample_count:
            temp_sample_count = base_sample_count
            temp_max_iter = temp_sample_count * epoch
            while temp_sample_count < sample_count:
                temp_sample_count = temp_sample_count + base_sample_count
                temp_max_iter = temp_max_iter + temp_sample_count * epoch
            if temp_max_iter == start_iter:
                pass_train = True
            if temp_max_iter < start_iter:
                raise Exception("主动学习恢复训练出现问题，之前手动修改过？")
        else:
            temp_max_iter = start_iter + sample_count * epoch
        # temp_max_iter = start_iter + 100
        if not (first_iter and pass_train):  # 如果是中间挑选样本的过程中出现问题就直接跳过训练而是直接取寻找样本
            trainer.temp_max_iter = temp_max_iter
            trainer.train(start_iter, cfg.train.max_iter)
        start_iter = temp_max_iter
        # 此处开始进行主动学习样本的筛选以及dataloader重载
        sample_count += base_sample_count
        geal_file_list = active_learning.read_from_file(geal_file_name)
        logger.info('当前轮次训练完毕，现在主动学习将选择出 {} 个样本进行训练,目前已经选出了 {} 个样本进行训练'.format(str(sample_count), str(len(geal_file_list))))
        select_sample_list = active_learning.geal_sampling(trainer.model, train_loader_all, sample_count, geal_file_list, sample_use_al, geal_file_name)
        active_learning.add_list_to_list(select_sample_list, geal_file_list)
        trainer.model.set_mode_sampling(False)
        logger.info('现在开始重新载入dataloader')
        trainer.reset_data_loader_local(instantiate(cfg.dataloader.train))
        first_iter = False


def main(args):
    cfg = LazyConfig.load(args.config_file)
    cfg = LazyConfig.apply_overrides(cfg, args.opts)
    default_setup(cfg, args)

    if args.eval_only:
        model = instantiate(cfg.model)
        model.to(cfg.train.device)
        model = create_ddp_model(model)
        DetectionCheckpointer(model).load(cfg.train.init_checkpoint)
        print(do_test(cfg, model))
    else:
        do_train(args, cfg)


if __name__ == "__main__":
    from detectron2.data import DatasetCatalog
    from detectron2.data.datasets import load_coco_json

    DatasetCatalog.register('coco_2017_train_fddance', lambda: load_coco_json(
        '/mnt/84BA4F3DBA4F2ACE/ubuntu/data/dataset/coco/annotations/instances_train2017.json',
        '/mnt/84BA4F3DBA4F2ACE/ubuntu/data/dataset/coco/train2017',
        'coco_2017_train_fddance'))
    DatasetCatalog.register('coco_2017_val_fddance', lambda: load_coco_json(
        '/mnt/84BA4F3DBA4F2ACE/ubuntu/data/dataset/coco/annotations/instances_val2017.json',
        '/mnt/84BA4F3DBA4F2ACE/ubuntu/data/dataset/coco/val2017',
        'coco_2017_val_fddance'))
    args = default_argument_parser(config_file='projects/dino/configs/dino_r50_4scale_12ep.py', resume=train_resume)
    # args.add_argument()
    args = args.parse_args()
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
