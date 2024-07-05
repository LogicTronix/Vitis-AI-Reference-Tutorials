# GENETARED BY NNDCT, DO NOT EDIT!

import torch
from torch import tensor
import pytorch_nndct as py_nndct

class DetectMultiBackend(py_nndct.nn.NndctQuantModel):
    def __init__(self):
        super(DetectMultiBackend, self).__init__()
        self.module_0 = py_nndct.nn.Input() #DetectMultiBackend::input_0
        self.module_1 = py_nndct.nn.Conv2d(in_channels=3, out_channels=48, kernel_size=[6, 6], stride=[2, 2], padding=[2, 2], dilation=[1, 1], groups=1, bias=True) #DetectMultiBackend::DetectMultiBackend/DetectionModel[model]/Conv[model]/Conv[0]/Conv2d[conv]/input.3
        self.module_2 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #DetectMultiBackend::DetectMultiBackend/DetectionModel[model]/Conv[model]/Conv[0]/LeakyReLU[act]/input.5
        self.module_3 = py_nndct.nn.Conv2d(in_channels=48, out_channels=96, kernel_size=[3, 3], stride=[2, 2], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #DetectMultiBackend::DetectMultiBackend/DetectionModel[model]/Conv[model]/Conv[1]/Conv2d[conv]/input.7
        self.module_4 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #DetectMultiBackend::DetectMultiBackend/DetectionModel[model]/Conv[model]/Conv[1]/LeakyReLU[act]/input.9
        self.module_5 = py_nndct.nn.Conv2d(in_channels=96, out_channels=48, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #DetectMultiBackend::DetectMultiBackend/DetectionModel[model]/C3[model]/C3[2]/Conv[cv1]/Conv2d[conv]/input.11
        self.module_6 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #DetectMultiBackend::DetectMultiBackend/DetectionModel[model]/C3[model]/C3[2]/Conv[cv1]/LeakyReLU[act]/input.13
        self.module_7 = py_nndct.nn.Conv2d(in_channels=48, out_channels=48, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #DetectMultiBackend::DetectMultiBackend/DetectionModel[model]/C3[model]/C3[2]/Sequential[m]/Bottleneck[0]/Conv[cv1]/Conv2d[conv]/input.15
        self.module_8 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #DetectMultiBackend::DetectMultiBackend/DetectionModel[model]/C3[model]/C3[2]/Sequential[m]/Bottleneck[0]/Conv[cv1]/LeakyReLU[act]/input.17
        self.module_9 = py_nndct.nn.Conv2d(in_channels=48, out_channels=48, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #DetectMultiBackend::DetectMultiBackend/DetectionModel[model]/C3[model]/C3[2]/Sequential[m]/Bottleneck[0]/Conv[cv2]/Conv2d[conv]/input.19
        self.module_10 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #DetectMultiBackend::DetectMultiBackend/DetectionModel[model]/C3[model]/C3[2]/Sequential[m]/Bottleneck[0]/Conv[cv2]/LeakyReLU[act]/11690
        self.module_11 = py_nndct.nn.Add() #DetectMultiBackend::DetectMultiBackend/DetectionModel[model]/C3[model]/C3[2]/Sequential[m]/Bottleneck[0]/input.21
        self.module_12 = py_nndct.nn.Conv2d(in_channels=48, out_channels=48, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #DetectMultiBackend::DetectMultiBackend/DetectionModel[model]/C3[model]/C3[2]/Sequential[m]/Bottleneck[1]/Conv[cv1]/Conv2d[conv]/input.23
        self.module_13 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #DetectMultiBackend::DetectMultiBackend/DetectionModel[model]/C3[model]/C3[2]/Sequential[m]/Bottleneck[1]/Conv[cv1]/LeakyReLU[act]/input.25
        self.module_14 = py_nndct.nn.Conv2d(in_channels=48, out_channels=48, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #DetectMultiBackend::DetectMultiBackend/DetectionModel[model]/C3[model]/C3[2]/Sequential[m]/Bottleneck[1]/Conv[cv2]/Conv2d[conv]/input.27
        self.module_15 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #DetectMultiBackend::DetectMultiBackend/DetectionModel[model]/C3[model]/C3[2]/Sequential[m]/Bottleneck[1]/Conv[cv2]/LeakyReLU[act]/11734
        self.module_16 = py_nndct.nn.Add() #DetectMultiBackend::DetectMultiBackend/DetectionModel[model]/C3[model]/C3[2]/Sequential[m]/Bottleneck[1]/11736
        self.module_17 = py_nndct.nn.Conv2d(in_channels=96, out_channels=48, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #DetectMultiBackend::DetectMultiBackend/DetectionModel[model]/C3[model]/C3[2]/Conv[cv2]/Conv2d[conv]/input.29
        self.module_18 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #DetectMultiBackend::DetectMultiBackend/DetectionModel[model]/C3[model]/C3[2]/Conv[cv2]/LeakyReLU[act]/11757
        self.module_19 = py_nndct.nn.Cat() #DetectMultiBackend::DetectMultiBackend/DetectionModel[model]/C3[model]/C3[2]/input.31
        self.module_20 = py_nndct.nn.Conv2d(in_channels=96, out_channels=96, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #DetectMultiBackend::DetectMultiBackend/DetectionModel[model]/C3[model]/C3[2]/Conv[cv3]/Conv2d[conv]/input.33
        self.module_21 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #DetectMultiBackend::DetectMultiBackend/DetectionModel[model]/C3[model]/C3[2]/Conv[cv3]/LeakyReLU[act]/input.35
        self.module_22 = py_nndct.nn.Conv2d(in_channels=96, out_channels=192, kernel_size=[3, 3], stride=[2, 2], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #DetectMultiBackend::DetectMultiBackend/DetectionModel[model]/Conv[model]/Conv[3]/Conv2d[conv]/input.37
        self.module_23 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #DetectMultiBackend::DetectMultiBackend/DetectionModel[model]/Conv[model]/Conv[3]/LeakyReLU[act]/input.39
        self.module_24 = py_nndct.nn.Conv2d(in_channels=192, out_channels=96, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #DetectMultiBackend::DetectMultiBackend/DetectionModel[model]/C3[model]/C3[4]/Conv[cv1]/Conv2d[conv]/input.41
        self.module_25 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #DetectMultiBackend::DetectMultiBackend/DetectionModel[model]/C3[model]/C3[4]/Conv[cv1]/LeakyReLU[act]/input.43
        self.module_26 = py_nndct.nn.Conv2d(in_channels=96, out_channels=96, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #DetectMultiBackend::DetectMultiBackend/DetectionModel[model]/C3[model]/C3[4]/Sequential[m]/Bottleneck[0]/Conv[cv1]/Conv2d[conv]/input.45
        self.module_27 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #DetectMultiBackend::DetectMultiBackend/DetectionModel[model]/C3[model]/C3[4]/Sequential[m]/Bottleneck[0]/Conv[cv1]/LeakyReLU[act]/input.47
        self.module_28 = py_nndct.nn.Conv2d(in_channels=96, out_channels=96, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #DetectMultiBackend::DetectMultiBackend/DetectionModel[model]/C3[model]/C3[4]/Sequential[m]/Bottleneck[0]/Conv[cv2]/Conv2d[conv]/input.49
        self.module_29 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #DetectMultiBackend::DetectMultiBackend/DetectionModel[model]/C3[model]/C3[4]/Sequential[m]/Bottleneck[0]/Conv[cv2]/LeakyReLU[act]/11865
        self.module_30 = py_nndct.nn.Add() #DetectMultiBackend::DetectMultiBackend/DetectionModel[model]/C3[model]/C3[4]/Sequential[m]/Bottleneck[0]/input.51
        self.module_31 = py_nndct.nn.Conv2d(in_channels=96, out_channels=96, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #DetectMultiBackend::DetectMultiBackend/DetectionModel[model]/C3[model]/C3[4]/Sequential[m]/Bottleneck[1]/Conv[cv1]/Conv2d[conv]/input.53
        self.module_32 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #DetectMultiBackend::DetectMultiBackend/DetectionModel[model]/C3[model]/C3[4]/Sequential[m]/Bottleneck[1]/Conv[cv1]/LeakyReLU[act]/input.55
        self.module_33 = py_nndct.nn.Conv2d(in_channels=96, out_channels=96, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #DetectMultiBackend::DetectMultiBackend/DetectionModel[model]/C3[model]/C3[4]/Sequential[m]/Bottleneck[1]/Conv[cv2]/Conv2d[conv]/input.57
        self.module_34 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #DetectMultiBackend::DetectMultiBackend/DetectionModel[model]/C3[model]/C3[4]/Sequential[m]/Bottleneck[1]/Conv[cv2]/LeakyReLU[act]/11909
        self.module_35 = py_nndct.nn.Add() #DetectMultiBackend::DetectMultiBackend/DetectionModel[model]/C3[model]/C3[4]/Sequential[m]/Bottleneck[1]/input.59
        self.module_36 = py_nndct.nn.Conv2d(in_channels=96, out_channels=96, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #DetectMultiBackend::DetectMultiBackend/DetectionModel[model]/C3[model]/C3[4]/Sequential[m]/Bottleneck[2]/Conv[cv1]/Conv2d[conv]/input.61
        self.module_37 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #DetectMultiBackend::DetectMultiBackend/DetectionModel[model]/C3[model]/C3[4]/Sequential[m]/Bottleneck[2]/Conv[cv1]/LeakyReLU[act]/input.63
        self.module_38 = py_nndct.nn.Conv2d(in_channels=96, out_channels=96, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #DetectMultiBackend::DetectMultiBackend/DetectionModel[model]/C3[model]/C3[4]/Sequential[m]/Bottleneck[2]/Conv[cv2]/Conv2d[conv]/input.65
        self.module_39 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #DetectMultiBackend::DetectMultiBackend/DetectionModel[model]/C3[model]/C3[4]/Sequential[m]/Bottleneck[2]/Conv[cv2]/LeakyReLU[act]/11953
        self.module_40 = py_nndct.nn.Add() #DetectMultiBackend::DetectMultiBackend/DetectionModel[model]/C3[model]/C3[4]/Sequential[m]/Bottleneck[2]/input.67
        self.module_41 = py_nndct.nn.Conv2d(in_channels=96, out_channels=96, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #DetectMultiBackend::DetectMultiBackend/DetectionModel[model]/C3[model]/C3[4]/Sequential[m]/Bottleneck[3]/Conv[cv1]/Conv2d[conv]/input.69
        self.module_42 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #DetectMultiBackend::DetectMultiBackend/DetectionModel[model]/C3[model]/C3[4]/Sequential[m]/Bottleneck[3]/Conv[cv1]/LeakyReLU[act]/input.71
        self.module_43 = py_nndct.nn.Conv2d(in_channels=96, out_channels=96, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #DetectMultiBackend::DetectMultiBackend/DetectionModel[model]/C3[model]/C3[4]/Sequential[m]/Bottleneck[3]/Conv[cv2]/Conv2d[conv]/input.73
        self.module_44 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #DetectMultiBackend::DetectMultiBackend/DetectionModel[model]/C3[model]/C3[4]/Sequential[m]/Bottleneck[3]/Conv[cv2]/LeakyReLU[act]/11997
        self.module_45 = py_nndct.nn.Add() #DetectMultiBackend::DetectMultiBackend/DetectionModel[model]/C3[model]/C3[4]/Sequential[m]/Bottleneck[3]/11999
        self.module_46 = py_nndct.nn.Conv2d(in_channels=192, out_channels=96, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #DetectMultiBackend::DetectMultiBackend/DetectionModel[model]/C3[model]/C3[4]/Conv[cv2]/Conv2d[conv]/input.75
        self.module_47 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #DetectMultiBackend::DetectMultiBackend/DetectionModel[model]/C3[model]/C3[4]/Conv[cv2]/LeakyReLU[act]/12020
        self.module_48 = py_nndct.nn.Cat() #DetectMultiBackend::DetectMultiBackend/DetectionModel[model]/C3[model]/C3[4]/input.77
        self.module_49 = py_nndct.nn.Conv2d(in_channels=192, out_channels=192, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #DetectMultiBackend::DetectMultiBackend/DetectionModel[model]/C3[model]/C3[4]/Conv[cv3]/Conv2d[conv]/input.79
        self.module_50 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #DetectMultiBackend::DetectMultiBackend/DetectionModel[model]/C3[model]/C3[4]/Conv[cv3]/LeakyReLU[act]/input.81
        self.module_51 = py_nndct.nn.Conv2d(in_channels=192, out_channels=384, kernel_size=[3, 3], stride=[2, 2], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #DetectMultiBackend::DetectMultiBackend/DetectionModel[model]/Conv[model]/Conv[5]/Conv2d[conv]/input.83
        self.module_52 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #DetectMultiBackend::DetectMultiBackend/DetectionModel[model]/Conv[model]/Conv[5]/LeakyReLU[act]/input.85
        self.module_53 = py_nndct.nn.Conv2d(in_channels=384, out_channels=192, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #DetectMultiBackend::DetectMultiBackend/DetectionModel[model]/C3[model]/C3[6]/Conv[cv1]/Conv2d[conv]/input.87
        self.module_54 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #DetectMultiBackend::DetectMultiBackend/DetectionModel[model]/C3[model]/C3[6]/Conv[cv1]/LeakyReLU[act]/input.89
        self.module_55 = py_nndct.nn.Conv2d(in_channels=192, out_channels=192, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #DetectMultiBackend::DetectMultiBackend/DetectionModel[model]/C3[model]/C3[6]/Sequential[m]/Bottleneck[0]/Conv[cv1]/Conv2d[conv]/input.91
        self.module_56 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #DetectMultiBackend::DetectMultiBackend/DetectionModel[model]/C3[model]/C3[6]/Sequential[m]/Bottleneck[0]/Conv[cv1]/LeakyReLU[act]/input.93
        self.module_57 = py_nndct.nn.Conv2d(in_channels=192, out_channels=192, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #DetectMultiBackend::DetectMultiBackend/DetectionModel[model]/C3[model]/C3[6]/Sequential[m]/Bottleneck[0]/Conv[cv2]/Conv2d[conv]/input.95
        self.module_58 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #DetectMultiBackend::DetectMultiBackend/DetectionModel[model]/C3[model]/C3[6]/Sequential[m]/Bottleneck[0]/Conv[cv2]/LeakyReLU[act]/12128
        self.module_59 = py_nndct.nn.Add() #DetectMultiBackend::DetectMultiBackend/DetectionModel[model]/C3[model]/C3[6]/Sequential[m]/Bottleneck[0]/input.97
        self.module_60 = py_nndct.nn.Conv2d(in_channels=192, out_channels=192, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #DetectMultiBackend::DetectMultiBackend/DetectionModel[model]/C3[model]/C3[6]/Sequential[m]/Bottleneck[1]/Conv[cv1]/Conv2d[conv]/input.99
        self.module_61 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #DetectMultiBackend::DetectMultiBackend/DetectionModel[model]/C3[model]/C3[6]/Sequential[m]/Bottleneck[1]/Conv[cv1]/LeakyReLU[act]/input.101
        self.module_62 = py_nndct.nn.Conv2d(in_channels=192, out_channels=192, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #DetectMultiBackend::DetectMultiBackend/DetectionModel[model]/C3[model]/C3[6]/Sequential[m]/Bottleneck[1]/Conv[cv2]/Conv2d[conv]/input.103
        self.module_63 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #DetectMultiBackend::DetectMultiBackend/DetectionModel[model]/C3[model]/C3[6]/Sequential[m]/Bottleneck[1]/Conv[cv2]/LeakyReLU[act]/12172
        self.module_64 = py_nndct.nn.Add() #DetectMultiBackend::DetectMultiBackend/DetectionModel[model]/C3[model]/C3[6]/Sequential[m]/Bottleneck[1]/input.105
        self.module_65 = py_nndct.nn.Conv2d(in_channels=192, out_channels=192, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #DetectMultiBackend::DetectMultiBackend/DetectionModel[model]/C3[model]/C3[6]/Sequential[m]/Bottleneck[2]/Conv[cv1]/Conv2d[conv]/input.107
        self.module_66 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #DetectMultiBackend::DetectMultiBackend/DetectionModel[model]/C3[model]/C3[6]/Sequential[m]/Bottleneck[2]/Conv[cv1]/LeakyReLU[act]/input.109
        self.module_67 = py_nndct.nn.Conv2d(in_channels=192, out_channels=192, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #DetectMultiBackend::DetectMultiBackend/DetectionModel[model]/C3[model]/C3[6]/Sequential[m]/Bottleneck[2]/Conv[cv2]/Conv2d[conv]/input.111
        self.module_68 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #DetectMultiBackend::DetectMultiBackend/DetectionModel[model]/C3[model]/C3[6]/Sequential[m]/Bottleneck[2]/Conv[cv2]/LeakyReLU[act]/12216
        self.module_69 = py_nndct.nn.Add() #DetectMultiBackend::DetectMultiBackend/DetectionModel[model]/C3[model]/C3[6]/Sequential[m]/Bottleneck[2]/input.113
        self.module_70 = py_nndct.nn.Conv2d(in_channels=192, out_channels=192, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #DetectMultiBackend::DetectMultiBackend/DetectionModel[model]/C3[model]/C3[6]/Sequential[m]/Bottleneck[3]/Conv[cv1]/Conv2d[conv]/input.115
        self.module_71 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #DetectMultiBackend::DetectMultiBackend/DetectionModel[model]/C3[model]/C3[6]/Sequential[m]/Bottleneck[3]/Conv[cv1]/LeakyReLU[act]/input.117
        self.module_72 = py_nndct.nn.Conv2d(in_channels=192, out_channels=192, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #DetectMultiBackend::DetectMultiBackend/DetectionModel[model]/C3[model]/C3[6]/Sequential[m]/Bottleneck[3]/Conv[cv2]/Conv2d[conv]/input.119
        self.module_73 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #DetectMultiBackend::DetectMultiBackend/DetectionModel[model]/C3[model]/C3[6]/Sequential[m]/Bottleneck[3]/Conv[cv2]/LeakyReLU[act]/12260
        self.module_74 = py_nndct.nn.Add() #DetectMultiBackend::DetectMultiBackend/DetectionModel[model]/C3[model]/C3[6]/Sequential[m]/Bottleneck[3]/input.121
        self.module_75 = py_nndct.nn.Conv2d(in_channels=192, out_channels=192, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #DetectMultiBackend::DetectMultiBackend/DetectionModel[model]/C3[model]/C3[6]/Sequential[m]/Bottleneck[4]/Conv[cv1]/Conv2d[conv]/input.123
        self.module_76 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #DetectMultiBackend::DetectMultiBackend/DetectionModel[model]/C3[model]/C3[6]/Sequential[m]/Bottleneck[4]/Conv[cv1]/LeakyReLU[act]/input.125
        self.module_77 = py_nndct.nn.Conv2d(in_channels=192, out_channels=192, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #DetectMultiBackend::DetectMultiBackend/DetectionModel[model]/C3[model]/C3[6]/Sequential[m]/Bottleneck[4]/Conv[cv2]/Conv2d[conv]/input.127
        self.module_78 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #DetectMultiBackend::DetectMultiBackend/DetectionModel[model]/C3[model]/C3[6]/Sequential[m]/Bottleneck[4]/Conv[cv2]/LeakyReLU[act]/12304
        self.module_79 = py_nndct.nn.Add() #DetectMultiBackend::DetectMultiBackend/DetectionModel[model]/C3[model]/C3[6]/Sequential[m]/Bottleneck[4]/input.129
        self.module_80 = py_nndct.nn.Conv2d(in_channels=192, out_channels=192, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #DetectMultiBackend::DetectMultiBackend/DetectionModel[model]/C3[model]/C3[6]/Sequential[m]/Bottleneck[5]/Conv[cv1]/Conv2d[conv]/input.131
        self.module_81 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #DetectMultiBackend::DetectMultiBackend/DetectionModel[model]/C3[model]/C3[6]/Sequential[m]/Bottleneck[5]/Conv[cv1]/LeakyReLU[act]/input.133
        self.module_82 = py_nndct.nn.Conv2d(in_channels=192, out_channels=192, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #DetectMultiBackend::DetectMultiBackend/DetectionModel[model]/C3[model]/C3[6]/Sequential[m]/Bottleneck[5]/Conv[cv2]/Conv2d[conv]/input.135
        self.module_83 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #DetectMultiBackend::DetectMultiBackend/DetectionModel[model]/C3[model]/C3[6]/Sequential[m]/Bottleneck[5]/Conv[cv2]/LeakyReLU[act]/12348
        self.module_84 = py_nndct.nn.Add() #DetectMultiBackend::DetectMultiBackend/DetectionModel[model]/C3[model]/C3[6]/Sequential[m]/Bottleneck[5]/12350
        self.module_85 = py_nndct.nn.Conv2d(in_channels=384, out_channels=192, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #DetectMultiBackend::DetectMultiBackend/DetectionModel[model]/C3[model]/C3[6]/Conv[cv2]/Conv2d[conv]/input.137
        self.module_86 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #DetectMultiBackend::DetectMultiBackend/DetectionModel[model]/C3[model]/C3[6]/Conv[cv2]/LeakyReLU[act]/12371
        self.module_87 = py_nndct.nn.Cat() #DetectMultiBackend::DetectMultiBackend/DetectionModel[model]/C3[model]/C3[6]/input.139
        self.module_88 = py_nndct.nn.Conv2d(in_channels=384, out_channels=384, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #DetectMultiBackend::DetectMultiBackend/DetectionModel[model]/C3[model]/C3[6]/Conv[cv3]/Conv2d[conv]/input.141
        self.module_89 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #DetectMultiBackend::DetectMultiBackend/DetectionModel[model]/C3[model]/C3[6]/Conv[cv3]/LeakyReLU[act]/input.143
        self.module_90 = py_nndct.nn.Conv2d(in_channels=384, out_channels=768, kernel_size=[3, 3], stride=[2, 2], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #DetectMultiBackend::DetectMultiBackend/DetectionModel[model]/Conv[model]/Conv[7]/Conv2d[conv]/input.145
        self.module_91 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #DetectMultiBackend::DetectMultiBackend/DetectionModel[model]/Conv[model]/Conv[7]/LeakyReLU[act]/input.147
        self.module_92 = py_nndct.nn.Conv2d(in_channels=768, out_channels=384, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #DetectMultiBackend::DetectMultiBackend/DetectionModel[model]/C3[model]/C3[8]/Conv[cv1]/Conv2d[conv]/input.149
        self.module_93 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #DetectMultiBackend::DetectMultiBackend/DetectionModel[model]/C3[model]/C3[8]/Conv[cv1]/LeakyReLU[act]/input.151
        self.module_94 = py_nndct.nn.Conv2d(in_channels=384, out_channels=384, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #DetectMultiBackend::DetectMultiBackend/DetectionModel[model]/C3[model]/C3[8]/Sequential[m]/Bottleneck[0]/Conv[cv1]/Conv2d[conv]/input.153
        self.module_95 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #DetectMultiBackend::DetectMultiBackend/DetectionModel[model]/C3[model]/C3[8]/Sequential[m]/Bottleneck[0]/Conv[cv1]/LeakyReLU[act]/input.155
        self.module_96 = py_nndct.nn.Conv2d(in_channels=384, out_channels=384, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #DetectMultiBackend::DetectMultiBackend/DetectionModel[model]/C3[model]/C3[8]/Sequential[m]/Bottleneck[0]/Conv[cv2]/Conv2d[conv]/input.157
        self.module_97 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #DetectMultiBackend::DetectMultiBackend/DetectionModel[model]/C3[model]/C3[8]/Sequential[m]/Bottleneck[0]/Conv[cv2]/LeakyReLU[act]/12479
        self.module_98 = py_nndct.nn.Add() #DetectMultiBackend::DetectMultiBackend/DetectionModel[model]/C3[model]/C3[8]/Sequential[m]/Bottleneck[0]/input.159
        self.module_99 = py_nndct.nn.Conv2d(in_channels=384, out_channels=384, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #DetectMultiBackend::DetectMultiBackend/DetectionModel[model]/C3[model]/C3[8]/Sequential[m]/Bottleneck[1]/Conv[cv1]/Conv2d[conv]/input.161
        self.module_100 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #DetectMultiBackend::DetectMultiBackend/DetectionModel[model]/C3[model]/C3[8]/Sequential[m]/Bottleneck[1]/Conv[cv1]/LeakyReLU[act]/input.163
        self.module_101 = py_nndct.nn.Conv2d(in_channels=384, out_channels=384, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #DetectMultiBackend::DetectMultiBackend/DetectionModel[model]/C3[model]/C3[8]/Sequential[m]/Bottleneck[1]/Conv[cv2]/Conv2d[conv]/input.165
        self.module_102 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #DetectMultiBackend::DetectMultiBackend/DetectionModel[model]/C3[model]/C3[8]/Sequential[m]/Bottleneck[1]/Conv[cv2]/LeakyReLU[act]/12523
        self.module_103 = py_nndct.nn.Add() #DetectMultiBackend::DetectMultiBackend/DetectionModel[model]/C3[model]/C3[8]/Sequential[m]/Bottleneck[1]/12525
        self.module_104 = py_nndct.nn.Conv2d(in_channels=768, out_channels=384, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #DetectMultiBackend::DetectMultiBackend/DetectionModel[model]/C3[model]/C3[8]/Conv[cv2]/Conv2d[conv]/input.167
        self.module_105 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #DetectMultiBackend::DetectMultiBackend/DetectionModel[model]/C3[model]/C3[8]/Conv[cv2]/LeakyReLU[act]/12546
        self.module_106 = py_nndct.nn.Cat() #DetectMultiBackend::DetectMultiBackend/DetectionModel[model]/C3[model]/C3[8]/input.169
        self.module_107 = py_nndct.nn.Conv2d(in_channels=768, out_channels=768, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #DetectMultiBackend::DetectMultiBackend/DetectionModel[model]/C3[model]/C3[8]/Conv[cv3]/Conv2d[conv]/input.171
        self.module_108 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #DetectMultiBackend::DetectMultiBackend/DetectionModel[model]/C3[model]/C3[8]/Conv[cv3]/LeakyReLU[act]/input.173
        self.module_109 = py_nndct.nn.Conv2d(in_channels=768, out_channels=384, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #DetectMultiBackend::DetectMultiBackend/DetectionModel[model]/SPPF[model]/SPPF[9]/Conv[cv1]/Conv2d[conv]/input.175
        self.module_110 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #DetectMultiBackend::DetectMultiBackend/DetectionModel[model]/SPPF[model]/SPPF[9]/Conv[cv1]/LeakyReLU[act]/12591
        self.module_111 = py_nndct.nn.MaxPool2d(kernel_size=[5, 5], stride=[1, 1], padding=[2, 2], dilation=[1, 1], ceil_mode=False) #DetectMultiBackend::DetectMultiBackend/DetectionModel[model]/SPPF[model]/SPPF[9]/MaxPool2d[m]/12605
        self.module_112 = py_nndct.nn.MaxPool2d(kernel_size=[5, 5], stride=[1, 1], padding=[2, 2], dilation=[1, 1], ceil_mode=False) #DetectMultiBackend::DetectMultiBackend/DetectionModel[model]/SPPF[model]/SPPF[9]/MaxPool2d[m]/12619
        self.module_113 = py_nndct.nn.MaxPool2d(kernel_size=[5, 5], stride=[1, 1], padding=[2, 2], dilation=[1, 1], ceil_mode=False) #DetectMultiBackend::DetectMultiBackend/DetectionModel[model]/SPPF[model]/SPPF[9]/MaxPool2d[m]/12633
        self.module_114 = py_nndct.nn.Cat() #DetectMultiBackend::DetectMultiBackend/DetectionModel[model]/SPPF[model]/SPPF[9]/input.177
        self.module_115 = py_nndct.nn.Conv2d(in_channels=1536, out_channels=768, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #DetectMultiBackend::DetectMultiBackend/DetectionModel[model]/SPPF[model]/SPPF[9]/Conv[cv2]/Conv2d[conv]/input.179
        self.module_116 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #DetectMultiBackend::DetectMultiBackend/DetectionModel[model]/SPPF[model]/SPPF[9]/Conv[cv2]/LeakyReLU[act]/input.181
        self.module_117 = py_nndct.nn.Conv2d(in_channels=768, out_channels=384, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #DetectMultiBackend::DetectMultiBackend/DetectionModel[model]/Conv[model]/Conv[10]/Conv2d[conv]/input.183
        self.module_118 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #DetectMultiBackend::DetectMultiBackend/DetectionModel[model]/Conv[model]/Conv[10]/LeakyReLU[act]/input.185
        self.module_119 = py_nndct.nn.Interpolate() #DetectMultiBackend::DetectMultiBackend/DetectionModel[model]/Upsample[model]/Upsample[11]/12683
        self.module_120 = py_nndct.nn.Cat() #DetectMultiBackend::DetectMultiBackend/DetectionModel[model]/Concat[model]/Concat[12]/input.187
        self.module_121 = py_nndct.nn.Conv2d(in_channels=768, out_channels=192, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #DetectMultiBackend::DetectMultiBackend/DetectionModel[model]/C3[model]/C3[13]/Conv[cv1]/Conv2d[conv]/input.189
        self.module_122 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #DetectMultiBackend::DetectMultiBackend/DetectionModel[model]/C3[model]/C3[13]/Conv[cv1]/LeakyReLU[act]/input.191
        self.module_123 = py_nndct.nn.Conv2d(in_channels=192, out_channels=192, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #DetectMultiBackend::DetectMultiBackend/DetectionModel[model]/C3[model]/C3[13]/Sequential[m]/Bottleneck[0]/Conv[cv1]/Conv2d[conv]/input.193
        self.module_124 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #DetectMultiBackend::DetectMultiBackend/DetectionModel[model]/C3[model]/C3[13]/Sequential[m]/Bottleneck[0]/Conv[cv1]/LeakyReLU[act]/input.195
        self.module_125 = py_nndct.nn.Conv2d(in_channels=192, out_channels=192, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #DetectMultiBackend::DetectMultiBackend/DetectionModel[model]/C3[model]/C3[13]/Sequential[m]/Bottleneck[0]/Conv[cv2]/Conv2d[conv]/input.197
        self.module_126 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #DetectMultiBackend::DetectMultiBackend/DetectionModel[model]/C3[model]/C3[13]/Sequential[m]/Bottleneck[0]/Conv[cv2]/LeakyReLU[act]/input.199
        self.module_127 = py_nndct.nn.Conv2d(in_channels=192, out_channels=192, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #DetectMultiBackend::DetectMultiBackend/DetectionModel[model]/C3[model]/C3[13]/Sequential[m]/Bottleneck[1]/Conv[cv1]/Conv2d[conv]/input.201
        self.module_128 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #DetectMultiBackend::DetectMultiBackend/DetectionModel[model]/C3[model]/C3[13]/Sequential[m]/Bottleneck[1]/Conv[cv1]/LeakyReLU[act]/input.203
        self.module_129 = py_nndct.nn.Conv2d(in_channels=192, out_channels=192, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #DetectMultiBackend::DetectMultiBackend/DetectionModel[model]/C3[model]/C3[13]/Sequential[m]/Bottleneck[1]/Conv[cv2]/Conv2d[conv]/input.205
        self.module_130 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #DetectMultiBackend::DetectMultiBackend/DetectionModel[model]/C3[model]/C3[13]/Sequential[m]/Bottleneck[1]/Conv[cv2]/LeakyReLU[act]/12791
        self.module_131 = py_nndct.nn.Conv2d(in_channels=768, out_channels=192, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #DetectMultiBackend::DetectMultiBackend/DetectionModel[model]/C3[model]/C3[13]/Conv[cv2]/Conv2d[conv]/input.207
        self.module_132 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #DetectMultiBackend::DetectMultiBackend/DetectionModel[model]/C3[model]/C3[13]/Conv[cv2]/LeakyReLU[act]/12812
        self.module_133 = py_nndct.nn.Cat() #DetectMultiBackend::DetectMultiBackend/DetectionModel[model]/C3[model]/C3[13]/input.209
        self.module_134 = py_nndct.nn.Conv2d(in_channels=384, out_channels=384, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #DetectMultiBackend::DetectMultiBackend/DetectionModel[model]/C3[model]/C3[13]/Conv[cv3]/Conv2d[conv]/input.211
        self.module_135 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #DetectMultiBackend::DetectMultiBackend/DetectionModel[model]/C3[model]/C3[13]/Conv[cv3]/LeakyReLU[act]/input.213
        self.module_136 = py_nndct.nn.Conv2d(in_channels=384, out_channels=192, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #DetectMultiBackend::DetectMultiBackend/DetectionModel[model]/Conv[model]/Conv[14]/Conv2d[conv]/input.215
        self.module_137 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #DetectMultiBackend::DetectMultiBackend/DetectionModel[model]/Conv[model]/Conv[14]/LeakyReLU[act]/input.217
        self.module_138 = py_nndct.nn.Interpolate() #DetectMultiBackend::DetectMultiBackend/DetectionModel[model]/Upsample[model]/Upsample[15]/12862
        self.module_139 = py_nndct.nn.Cat() #DetectMultiBackend::DetectMultiBackend/DetectionModel[model]/Concat[model]/Concat[16]/input.219
        self.module_140 = py_nndct.nn.Conv2d(in_channels=384, out_channels=96, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #DetectMultiBackend::DetectMultiBackend/DetectionModel[model]/C3[model]/C3[17]/Conv[cv1]/Conv2d[conv]/input.221
        self.module_141 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #DetectMultiBackend::DetectMultiBackend/DetectionModel[model]/C3[model]/C3[17]/Conv[cv1]/LeakyReLU[act]/input.223
        self.module_142 = py_nndct.nn.Conv2d(in_channels=96, out_channels=96, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #DetectMultiBackend::DetectMultiBackend/DetectionModel[model]/C3[model]/C3[17]/Sequential[m]/Bottleneck[0]/Conv[cv1]/Conv2d[conv]/input.225
        self.module_143 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #DetectMultiBackend::DetectMultiBackend/DetectionModel[model]/C3[model]/C3[17]/Sequential[m]/Bottleneck[0]/Conv[cv1]/LeakyReLU[act]/input.227
        self.module_144 = py_nndct.nn.Conv2d(in_channels=96, out_channels=96, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #DetectMultiBackend::DetectMultiBackend/DetectionModel[model]/C3[model]/C3[17]/Sequential[m]/Bottleneck[0]/Conv[cv2]/Conv2d[conv]/input.229
        self.module_145 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #DetectMultiBackend::DetectMultiBackend/DetectionModel[model]/C3[model]/C3[17]/Sequential[m]/Bottleneck[0]/Conv[cv2]/LeakyReLU[act]/input.231
        self.module_146 = py_nndct.nn.Conv2d(in_channels=96, out_channels=96, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #DetectMultiBackend::DetectMultiBackend/DetectionModel[model]/C3[model]/C3[17]/Sequential[m]/Bottleneck[1]/Conv[cv1]/Conv2d[conv]/input.233
        self.module_147 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #DetectMultiBackend::DetectMultiBackend/DetectionModel[model]/C3[model]/C3[17]/Sequential[m]/Bottleneck[1]/Conv[cv1]/LeakyReLU[act]/input.235
        self.module_148 = py_nndct.nn.Conv2d(in_channels=96, out_channels=96, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #DetectMultiBackend::DetectMultiBackend/DetectionModel[model]/C3[model]/C3[17]/Sequential[m]/Bottleneck[1]/Conv[cv2]/Conv2d[conv]/input.237
        self.module_149 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #DetectMultiBackend::DetectMultiBackend/DetectionModel[model]/C3[model]/C3[17]/Sequential[m]/Bottleneck[1]/Conv[cv2]/LeakyReLU[act]/12970
        self.module_150 = py_nndct.nn.Conv2d(in_channels=384, out_channels=96, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #DetectMultiBackend::DetectMultiBackend/DetectionModel[model]/C3[model]/C3[17]/Conv[cv2]/Conv2d[conv]/input.239
        self.module_151 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #DetectMultiBackend::DetectMultiBackend/DetectionModel[model]/C3[model]/C3[17]/Conv[cv2]/LeakyReLU[act]/12991
        self.module_152 = py_nndct.nn.Cat() #DetectMultiBackend::DetectMultiBackend/DetectionModel[model]/C3[model]/C3[17]/input.241
        self.module_153 = py_nndct.nn.Conv2d(in_channels=192, out_channels=192, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #DetectMultiBackend::DetectMultiBackend/DetectionModel[model]/C3[model]/C3[17]/Conv[cv3]/Conv2d[conv]/input.243
        self.module_154 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #DetectMultiBackend::DetectMultiBackend/DetectionModel[model]/C3[model]/C3[17]/Conv[cv3]/LeakyReLU[act]/input.245
        self.module_155 = py_nndct.nn.Conv2d(in_channels=192, out_channels=192, kernel_size=[3, 3], stride=[2, 2], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #DetectMultiBackend::DetectMultiBackend/DetectionModel[model]/Conv[model]/Conv[18]/Conv2d[conv]/input.247
        self.module_156 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #DetectMultiBackend::DetectMultiBackend/DetectionModel[model]/Conv[model]/Conv[18]/LeakyReLU[act]/13036
        self.module_157 = py_nndct.nn.Cat() #DetectMultiBackend::DetectMultiBackend/DetectionModel[model]/Concat[model]/Concat[19]/input.249
        self.module_158 = py_nndct.nn.Conv2d(in_channels=384, out_channels=192, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #DetectMultiBackend::DetectMultiBackend/DetectionModel[model]/C3[model]/C3[20]/Conv[cv1]/Conv2d[conv]/input.251
        self.module_159 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #DetectMultiBackend::DetectMultiBackend/DetectionModel[model]/C3[model]/C3[20]/Conv[cv1]/LeakyReLU[act]/input.253
        self.module_160 = py_nndct.nn.Conv2d(in_channels=192, out_channels=192, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #DetectMultiBackend::DetectMultiBackend/DetectionModel[model]/C3[model]/C3[20]/Sequential[m]/Bottleneck[0]/Conv[cv1]/Conv2d[conv]/input.255
        self.module_161 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #DetectMultiBackend::DetectMultiBackend/DetectionModel[model]/C3[model]/C3[20]/Sequential[m]/Bottleneck[0]/Conv[cv1]/LeakyReLU[act]/input.257
        self.module_162 = py_nndct.nn.Conv2d(in_channels=192, out_channels=192, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #DetectMultiBackend::DetectMultiBackend/DetectionModel[model]/C3[model]/C3[20]/Sequential[m]/Bottleneck[0]/Conv[cv2]/Conv2d[conv]/input.259
        self.module_163 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #DetectMultiBackend::DetectMultiBackend/DetectionModel[model]/C3[model]/C3[20]/Sequential[m]/Bottleneck[0]/Conv[cv2]/LeakyReLU[act]/input.261
        self.module_164 = py_nndct.nn.Conv2d(in_channels=192, out_channels=192, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #DetectMultiBackend::DetectMultiBackend/DetectionModel[model]/C3[model]/C3[20]/Sequential[m]/Bottleneck[1]/Conv[cv1]/Conv2d[conv]/input.263
        self.module_165 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #DetectMultiBackend::DetectMultiBackend/DetectionModel[model]/C3[model]/C3[20]/Sequential[m]/Bottleneck[1]/Conv[cv1]/LeakyReLU[act]/input.265
        self.module_166 = py_nndct.nn.Conv2d(in_channels=192, out_channels=192, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #DetectMultiBackend::DetectMultiBackend/DetectionModel[model]/C3[model]/C3[20]/Sequential[m]/Bottleneck[1]/Conv[cv2]/Conv2d[conv]/input.267
        self.module_167 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #DetectMultiBackend::DetectMultiBackend/DetectionModel[model]/C3[model]/C3[20]/Sequential[m]/Bottleneck[1]/Conv[cv2]/LeakyReLU[act]/13144
        self.module_168 = py_nndct.nn.Conv2d(in_channels=384, out_channels=192, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #DetectMultiBackend::DetectMultiBackend/DetectionModel[model]/C3[model]/C3[20]/Conv[cv2]/Conv2d[conv]/input.269
        self.module_169 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #DetectMultiBackend::DetectMultiBackend/DetectionModel[model]/C3[model]/C3[20]/Conv[cv2]/LeakyReLU[act]/13165
        self.module_170 = py_nndct.nn.Cat() #DetectMultiBackend::DetectMultiBackend/DetectionModel[model]/C3[model]/C3[20]/input.271
        self.module_171 = py_nndct.nn.Conv2d(in_channels=384, out_channels=384, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #DetectMultiBackend::DetectMultiBackend/DetectionModel[model]/C3[model]/C3[20]/Conv[cv3]/Conv2d[conv]/input.273
        self.module_172 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #DetectMultiBackend::DetectMultiBackend/DetectionModel[model]/C3[model]/C3[20]/Conv[cv3]/LeakyReLU[act]/input.275
        self.module_173 = py_nndct.nn.Conv2d(in_channels=384, out_channels=384, kernel_size=[3, 3], stride=[2, 2], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #DetectMultiBackend::DetectMultiBackend/DetectionModel[model]/Conv[model]/Conv[21]/Conv2d[conv]/input.277
        self.module_174 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #DetectMultiBackend::DetectMultiBackend/DetectionModel[model]/Conv[model]/Conv[21]/LeakyReLU[act]/13210
        self.module_175 = py_nndct.nn.Cat() #DetectMultiBackend::DetectMultiBackend/DetectionModel[model]/Concat[model]/Concat[22]/input.279
        self.module_176 = py_nndct.nn.Conv2d(in_channels=768, out_channels=384, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #DetectMultiBackend::DetectMultiBackend/DetectionModel[model]/C3[model]/C3[23]/Conv[cv1]/Conv2d[conv]/input.281
        self.module_177 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #DetectMultiBackend::DetectMultiBackend/DetectionModel[model]/C3[model]/C3[23]/Conv[cv1]/LeakyReLU[act]/input.283
        self.module_178 = py_nndct.nn.Conv2d(in_channels=384, out_channels=384, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #DetectMultiBackend::DetectMultiBackend/DetectionModel[model]/C3[model]/C3[23]/Sequential[m]/Bottleneck[0]/Conv[cv1]/Conv2d[conv]/input.285
        self.module_179 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #DetectMultiBackend::DetectMultiBackend/DetectionModel[model]/C3[model]/C3[23]/Sequential[m]/Bottleneck[0]/Conv[cv1]/LeakyReLU[act]/input.287
        self.module_180 = py_nndct.nn.Conv2d(in_channels=384, out_channels=384, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #DetectMultiBackend::DetectMultiBackend/DetectionModel[model]/C3[model]/C3[23]/Sequential[m]/Bottleneck[0]/Conv[cv2]/Conv2d[conv]/input.289
        self.module_181 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #DetectMultiBackend::DetectMultiBackend/DetectionModel[model]/C3[model]/C3[23]/Sequential[m]/Bottleneck[0]/Conv[cv2]/LeakyReLU[act]/input.291
        self.module_182 = py_nndct.nn.Conv2d(in_channels=384, out_channels=384, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #DetectMultiBackend::DetectMultiBackend/DetectionModel[model]/C3[model]/C3[23]/Sequential[m]/Bottleneck[1]/Conv[cv1]/Conv2d[conv]/input.293
        self.module_183 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #DetectMultiBackend::DetectMultiBackend/DetectionModel[model]/C3[model]/C3[23]/Sequential[m]/Bottleneck[1]/Conv[cv1]/LeakyReLU[act]/input.295
        self.module_184 = py_nndct.nn.Conv2d(in_channels=384, out_channels=384, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #DetectMultiBackend::DetectMultiBackend/DetectionModel[model]/C3[model]/C3[23]/Sequential[m]/Bottleneck[1]/Conv[cv2]/Conv2d[conv]/input.297
        self.module_185 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #DetectMultiBackend::DetectMultiBackend/DetectionModel[model]/C3[model]/C3[23]/Sequential[m]/Bottleneck[1]/Conv[cv2]/LeakyReLU[act]/13318
        self.module_186 = py_nndct.nn.Conv2d(in_channels=768, out_channels=384, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #DetectMultiBackend::DetectMultiBackend/DetectionModel[model]/C3[model]/C3[23]/Conv[cv2]/Conv2d[conv]/input.299
        self.module_187 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #DetectMultiBackend::DetectMultiBackend/DetectionModel[model]/C3[model]/C3[23]/Conv[cv2]/LeakyReLU[act]/13339
        self.module_188 = py_nndct.nn.Cat() #DetectMultiBackend::DetectMultiBackend/DetectionModel[model]/C3[model]/C3[23]/input.301
        self.module_189 = py_nndct.nn.Conv2d(in_channels=768, out_channels=768, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #DetectMultiBackend::DetectMultiBackend/DetectionModel[model]/C3[model]/C3[23]/Conv[cv3]/Conv2d[conv]/input.303
        self.module_190 = py_nndct.nn.LeakyReLU(negative_slope=0.1015625, inplace=True) #DetectMultiBackend::DetectMultiBackend/DetectionModel[model]/C3[model]/C3[23]/Conv[cv3]/LeakyReLU[act]/input
        self.module_191 = py_nndct.nn.Conv2d(in_channels=192, out_channels=54, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #DetectMultiBackend::DetectMultiBackend/DetectionModel[model]/Detect[model]/Detect[24]/Conv2d[m]/ModuleList[0]/13382
        self.module_192 = py_nndct.nn.Conv2d(in_channels=384, out_channels=54, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #DetectMultiBackend::DetectMultiBackend/DetectionModel[model]/Detect[model]/Detect[24]/Conv2d[m]/ModuleList[1]/13413
        self.module_193 = py_nndct.nn.Conv2d(in_channels=768, out_channels=54, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1, bias=True) #DetectMultiBackend::DetectMultiBackend/DetectionModel[model]/Detect[model]/Detect[24]/Conv2d[m]/ModuleList[2]/13444

    @py_nndct.nn.forward_processor
    def forward(self, *args):
        output_module_0 = self.module_0(input=args[0])
        output_module_0 = self.module_1(output_module_0)
        output_module_0 = self.module_2(output_module_0)
        output_module_0 = self.module_3(output_module_0)
        output_module_0 = self.module_4(output_module_0)
        output_module_5 = self.module_5(output_module_0)
        output_module_5 = self.module_6(output_module_5)
        output_module_7 = self.module_7(output_module_5)
        output_module_7 = self.module_8(output_module_7)
        output_module_7 = self.module_9(output_module_7)
        output_module_7 = self.module_10(output_module_7)
        output_module_11 = self.module_11(input=output_module_5, other=output_module_7, alpha=1)
        output_module_12 = self.module_12(output_module_11)
        output_module_12 = self.module_13(output_module_12)
        output_module_12 = self.module_14(output_module_12)
        output_module_12 = self.module_15(output_module_12)
        output_module_16 = self.module_16(input=output_module_11, other=output_module_12, alpha=1)
        output_module_17 = self.module_17(output_module_0)
        output_module_17 = self.module_18(output_module_17)
        output_module_16 = self.module_19(dim=1, tensors=[output_module_16,output_module_17])
        output_module_16 = self.module_20(output_module_16)
        output_module_16 = self.module_21(output_module_16)
        output_module_16 = self.module_22(output_module_16)
        output_module_16 = self.module_23(output_module_16)
        output_module_24 = self.module_24(output_module_16)
        output_module_24 = self.module_25(output_module_24)
        output_module_26 = self.module_26(output_module_24)
        output_module_26 = self.module_27(output_module_26)
        output_module_26 = self.module_28(output_module_26)
        output_module_26 = self.module_29(output_module_26)
        output_module_30 = self.module_30(input=output_module_24, other=output_module_26, alpha=1)
        output_module_31 = self.module_31(output_module_30)
        output_module_31 = self.module_32(output_module_31)
        output_module_31 = self.module_33(output_module_31)
        output_module_31 = self.module_34(output_module_31)
        output_module_35 = self.module_35(input=output_module_30, other=output_module_31, alpha=1)
        output_module_36 = self.module_36(output_module_35)
        output_module_36 = self.module_37(output_module_36)
        output_module_36 = self.module_38(output_module_36)
        output_module_36 = self.module_39(output_module_36)
        output_module_40 = self.module_40(input=output_module_35, other=output_module_36, alpha=1)
        output_module_41 = self.module_41(output_module_40)
        output_module_41 = self.module_42(output_module_41)
        output_module_41 = self.module_43(output_module_41)
        output_module_41 = self.module_44(output_module_41)
        output_module_45 = self.module_45(input=output_module_40, other=output_module_41, alpha=1)
        output_module_46 = self.module_46(output_module_16)
        output_module_46 = self.module_47(output_module_46)
        output_module_45 = self.module_48(dim=1, tensors=[output_module_45,output_module_46])
        output_module_45 = self.module_49(output_module_45)
        output_module_45 = self.module_50(output_module_45)
        output_module_51 = self.module_51(output_module_45)
        output_module_51 = self.module_52(output_module_51)
        output_module_53 = self.module_53(output_module_51)
        output_module_53 = self.module_54(output_module_53)
        output_module_55 = self.module_55(output_module_53)
        output_module_55 = self.module_56(output_module_55)
        output_module_55 = self.module_57(output_module_55)
        output_module_55 = self.module_58(output_module_55)
        output_module_59 = self.module_59(input=output_module_53, other=output_module_55, alpha=1)
        output_module_60 = self.module_60(output_module_59)
        output_module_60 = self.module_61(output_module_60)
        output_module_60 = self.module_62(output_module_60)
        output_module_60 = self.module_63(output_module_60)
        output_module_64 = self.module_64(input=output_module_59, other=output_module_60, alpha=1)
        output_module_65 = self.module_65(output_module_64)
        output_module_65 = self.module_66(output_module_65)
        output_module_65 = self.module_67(output_module_65)
        output_module_65 = self.module_68(output_module_65)
        output_module_69 = self.module_69(input=output_module_64, other=output_module_65, alpha=1)
        output_module_70 = self.module_70(output_module_69)
        output_module_70 = self.module_71(output_module_70)
        output_module_70 = self.module_72(output_module_70)
        output_module_70 = self.module_73(output_module_70)
        output_module_74 = self.module_74(input=output_module_69, other=output_module_70, alpha=1)
        output_module_75 = self.module_75(output_module_74)
        output_module_75 = self.module_76(output_module_75)
        output_module_75 = self.module_77(output_module_75)
        output_module_75 = self.module_78(output_module_75)
        output_module_79 = self.module_79(input=output_module_74, other=output_module_75, alpha=1)
        output_module_80 = self.module_80(output_module_79)
        output_module_80 = self.module_81(output_module_80)
        output_module_80 = self.module_82(output_module_80)
        output_module_80 = self.module_83(output_module_80)
        output_module_84 = self.module_84(input=output_module_79, other=output_module_80, alpha=1)
        output_module_85 = self.module_85(output_module_51)
        output_module_85 = self.module_86(output_module_85)
        output_module_84 = self.module_87(dim=1, tensors=[output_module_84,output_module_85])
        output_module_84 = self.module_88(output_module_84)
        output_module_84 = self.module_89(output_module_84)
        output_module_90 = self.module_90(output_module_84)
        output_module_90 = self.module_91(output_module_90)
        output_module_92 = self.module_92(output_module_90)
        output_module_92 = self.module_93(output_module_92)
        output_module_94 = self.module_94(output_module_92)
        output_module_94 = self.module_95(output_module_94)
        output_module_94 = self.module_96(output_module_94)
        output_module_94 = self.module_97(output_module_94)
        output_module_98 = self.module_98(input=output_module_92, other=output_module_94, alpha=1)
        output_module_99 = self.module_99(output_module_98)
        output_module_99 = self.module_100(output_module_99)
        output_module_99 = self.module_101(output_module_99)
        output_module_99 = self.module_102(output_module_99)
        output_module_103 = self.module_103(input=output_module_98, other=output_module_99, alpha=1)
        output_module_104 = self.module_104(output_module_90)
        output_module_104 = self.module_105(output_module_104)
        output_module_103 = self.module_106(dim=1, tensors=[output_module_103,output_module_104])
        output_module_103 = self.module_107(output_module_103)
        output_module_103 = self.module_108(output_module_103)
        output_module_103 = self.module_109(output_module_103)
        output_module_103 = self.module_110(output_module_103)
        output_module_111 = self.module_111(output_module_103)
        output_module_112 = self.module_112(output_module_111)
        output_module_113 = self.module_113(output_module_112)
        output_module_114 = self.module_114(dim=1, tensors=[output_module_103,output_module_111,output_module_112,output_module_113])
        output_module_114 = self.module_115(output_module_114)
        output_module_114 = self.module_116(output_module_114)
        output_module_114 = self.module_117(output_module_114)
        output_module_114 = self.module_118(output_module_114)
        output_module_119 = self.module_119(input=output_module_114, size=None, scale_factor=[2.0,2.0], mode='nearest')
        output_module_119 = self.module_120(dim=1, tensors=[output_module_119,output_module_84])
        output_module_121 = self.module_121(output_module_119)
        output_module_121 = self.module_122(output_module_121)
        output_module_121 = self.module_123(output_module_121)
        output_module_121 = self.module_124(output_module_121)
        output_module_121 = self.module_125(output_module_121)
        output_module_121 = self.module_126(output_module_121)
        output_module_121 = self.module_127(output_module_121)
        output_module_121 = self.module_128(output_module_121)
        output_module_121 = self.module_129(output_module_121)
        output_module_121 = self.module_130(output_module_121)
        output_module_131 = self.module_131(output_module_119)
        output_module_131 = self.module_132(output_module_131)
        output_module_121 = self.module_133(dim=1, tensors=[output_module_121,output_module_131])
        output_module_121 = self.module_134(output_module_121)
        output_module_121 = self.module_135(output_module_121)
        output_module_121 = self.module_136(output_module_121)
        output_module_121 = self.module_137(output_module_121)
        output_module_138 = self.module_138(input=output_module_121, size=None, scale_factor=[2.0,2.0], mode='nearest')
        output_module_138 = self.module_139(dim=1, tensors=[output_module_138,output_module_45])
        output_module_140 = self.module_140(output_module_138)
        output_module_140 = self.module_141(output_module_140)
        output_module_140 = self.module_142(output_module_140)
        output_module_140 = self.module_143(output_module_140)
        output_module_140 = self.module_144(output_module_140)
        output_module_140 = self.module_145(output_module_140)
        output_module_140 = self.module_146(output_module_140)
        output_module_140 = self.module_147(output_module_140)
        output_module_140 = self.module_148(output_module_140)
        output_module_140 = self.module_149(output_module_140)
        output_module_150 = self.module_150(output_module_138)
        output_module_150 = self.module_151(output_module_150)
        output_module_140 = self.module_152(dim=1, tensors=[output_module_140,output_module_150])
        output_module_140 = self.module_153(output_module_140)
        output_module_140 = self.module_154(output_module_140)
        output_module_155 = self.module_155(output_module_140)
        output_module_155 = self.module_156(output_module_155)
        output_module_155 = self.module_157(dim=1, tensors=[output_module_155,output_module_121])
        output_module_158 = self.module_158(output_module_155)
        output_module_158 = self.module_159(output_module_158)
        output_module_158 = self.module_160(output_module_158)
        output_module_158 = self.module_161(output_module_158)
        output_module_158 = self.module_162(output_module_158)
        output_module_158 = self.module_163(output_module_158)
        output_module_158 = self.module_164(output_module_158)
        output_module_158 = self.module_165(output_module_158)
        output_module_158 = self.module_166(output_module_158)
        output_module_158 = self.module_167(output_module_158)
        output_module_168 = self.module_168(output_module_155)
        output_module_168 = self.module_169(output_module_168)
        output_module_158 = self.module_170(dim=1, tensors=[output_module_158,output_module_168])
        output_module_158 = self.module_171(output_module_158)
        output_module_158 = self.module_172(output_module_158)
        output_module_173 = self.module_173(output_module_158)
        output_module_173 = self.module_174(output_module_173)
        output_module_173 = self.module_175(dim=1, tensors=[output_module_173,output_module_114])
        output_module_176 = self.module_176(output_module_173)
        output_module_176 = self.module_177(output_module_176)
        output_module_176 = self.module_178(output_module_176)
        output_module_176 = self.module_179(output_module_176)
        output_module_176 = self.module_180(output_module_176)
        output_module_176 = self.module_181(output_module_176)
        output_module_176 = self.module_182(output_module_176)
        output_module_176 = self.module_183(output_module_176)
        output_module_176 = self.module_184(output_module_176)
        output_module_176 = self.module_185(output_module_176)
        output_module_186 = self.module_186(output_module_173)
        output_module_186 = self.module_187(output_module_186)
        output_module_176 = self.module_188(dim=1, tensors=[output_module_176,output_module_186])
        output_module_176 = self.module_189(output_module_176)
        output_module_176 = self.module_190(output_module_176)
        output_module_191 = self.module_191(output_module_140)
        output_module_192 = self.module_192(output_module_158)
        output_module_176 = self.module_193(output_module_176)
        return (output_module_191,output_module_192,output_module_176)
