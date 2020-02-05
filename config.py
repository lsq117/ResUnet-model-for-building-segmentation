#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
  @Email:  guangmingwu2010@gmail.com
  @Copyright: go-hiroaki
  @License: MIT
"""
from utils.mass_datasets import *


class MASSdataset(object):
    """
        Dataset setting for Massachusetts
    """

    def __init__(self):
        self.train = msBD("Train", "train")
        self.val = msBD("Validation", "validation")
        self.test = msBD("Test", "test")
        self.in_ch = 3
        self.out_ch = self.train.nb_class


class LSSubdataset(object):
    """
        Dataset setting for NewZealand
    """

    def __init__(self):
        self.train8xsub = nzLS8xsub("nz-train-slc", "train")
        self.train = nzLS("nz-train-slc", "train")
        self.val = nzLS("nz-train-slc", "val")
        self.test = nzLS("nz-test-slc", "all")
        self.in_ch = 3
        self.out_ch = self.train.nb_class


class LSEdataset(object):
    """
        Dataset setting for NewZealand
    """

    def __init__(self):
        self.trainLSE = nzLSE("nz-train-slc", "train")
        self.train = nzLS("nz-train-slc", "train")
        self.val = nzLS("nz-train-slc", "val")
        self.test = nzLS("nz-test-slc", "all")
        self.in_ch = 3
        self.out_ch = self.train.nb_class


if __name__ == "__main__":
    print("Hello world")
