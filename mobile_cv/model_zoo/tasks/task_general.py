#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from mobile_cv.model_zoo.datasets import dataset_factory
from mobile_cv.model_zoo.models import model_zoo_factory
from mobile_cv.model_zoo.tasks.task_common import TaskCommon

from . import task_factory


@task_factory.TASK_FACTORY.register("general")
class TaskGeneral(TaskCommon):
    def __init__(self, model_args, dataset_args):
        super().__init__()
        self.model_args = model_args
        self.dataset_args = dataset_args

    def get_model(self):
        func = model_zoo_factory.get_model
        if isinstance(self.model_args, dict):
            return func(**self.model_args)
        elif isinstance(self.model_args, (list, tuple)):
            return func(*self.model_args)
        else:
            return func(self.model_args)

    def get_dataloader(self):
        func = dataset_factory.get
        if isinstance(self.dataset_args, dict):
            return func(**self.dataset_args)
        elif isinstance(self.dataset_args, (list, tuple)):
            return func(*self.dataset_args)
        else:
            return func(self.dataset_args)
