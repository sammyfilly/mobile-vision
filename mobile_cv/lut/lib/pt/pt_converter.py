#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import typing

import torch
from mobile_cv.common.misc.registry import Registry

from mobile_cv.lut.lib import lut_ops, lut_schema


PT_CONVERTER = Registry("pytorch_converter")


def get_module_name(m):
    return m.__class__.__name__


@PT_CONVERTER.register("EqualLinear")
def convert_EqualLinear(m, input_shapes):
    op = lut_ops.Linear(m.in_features, m.out_features, m.bias is not None)
    return lut_schema.OpInfo(op, input_shapes)


@PT_CONVERTER.register("ModulatedConv2d")
def convert_ModulatedConv2d(m, input_shapes):
    dilation = 1
    groups = 1
    bias = None
    if m.upsample:
        output_padding = 0
        op = lut_ops.ConvTranspose2d(
            m.in_channels,
            m.out_channels,
            m.kernel_size,
            m.stride,
            m.padding,
            output_padding,
            groups,
            bias,
            dilation,
        )
    else:
        op = lut_ops.Conv2d(
            m.in_channels,
            m.out_channels,
            m.kernel_size,
            m.stride,
            m.padding,
            dilation,
            groups,
            bias,
        )
    return lut_schema.OpInfo(op, input_shapes)


@PT_CONVERTER.register("Conv2d")
def convert_Conv2d(m, input_shapes):
    op = lut_ops.Conv2d(
        m.in_channels,
        m.out_channels,
        m.kernel_size,
        m.stride,
        m.padding,
        m.dilation,
        m.groups,
        m.bias is not None,
    )
    return lut_schema.OpInfo(op, input_shapes)


@PT_CONVERTER.register("Conv1d")
def convert_Conv1d(m, input_shapes):
    op = lut_ops.Conv1d(
        m.in_channels,
        m.out_channels,
        m.kernel_size,
        m.stride,
        m.padding,
        m.dilation,
        m.groups,
        m.bias is not None,
    )
    return lut_schema.OpInfo(op, input_shapes)


@PT_CONVERTER.register("Conv3d")
def convert_Conv3d(m, input_shapes):
    op = lut_ops.Conv3d(
        m.in_channels,
        m.out_channels,
        m.kernel_size,
        m.stride,
        m.padding,
        m.dilation,
        m.groups,
        m.bias is not None,
    )
    return lut_schema.OpInfo(op, input_shapes)


@PT_CONVERTER.register("ConvTranspose2d")
def convert_ConvTranspose2d(m, input_shapes):
    op = lut_ops.ConvTranspose2d(
        m.in_channels,
        m.out_channels,
        m.kernel_size,
        m.stride,
        m.padding,
        m.output_padding,
        m.groups,
        m.bias is not None,
        m.dilation,
    )
    return lut_schema.OpInfo(op, input_shapes)


@PT_CONVERTER.register("Linear")
def convert_Linear(m, input_shapes):
    op = lut_ops.Linear(m.in_features, m.out_features, m.bias is not None)
    return lut_schema.OpInfo(op, input_shapes)


@PT_CONVERTER.register("AdaptiveAvgPool2d")
def convert_AdaptiveAvgPool2d(m, input_shapes):
    op = lut_ops.AdaptiveAvgPool2d(m.output_size)
    return lut_schema.OpInfo(op, input_shapes)


@PT_CONVERTER.register("MatMul")
def convert_MatMul(m, input_shapes):
    op = lut_ops.MatMul()
    return lut_schema.OpInfo(op, input_shapes)


@PT_CONVERTER.register("MultiheadAttention")
def convert_MultiheadAttention(m, input_shapes):
    op = lut_ops.MultiheadAttention(
        m.embed_dim,
        m.num_heads,
        kdim=m.kdim,
        vdim=m.vdim,
    )
    return lut_schema.OpInfo(op, input_shapes)


def convert_module(m: torch.nn.Module, shape):
    name = get_module_name(m)
    func = PT_CONVERTER.get(name, is_raise=False)
    return func(m, shape) if func is not None else None


def convert_all_modules(model: torch.nn.Module, get_module_shape: typing.Callable):
    ret = []

    def _convert(m):
        shapes = get_module_shape(m)
        if shapes is not None:
            cur = convert_module(m, shapes)
            if cur is not None:
                ret.append(cur)

    model.apply(_convert)

    return ret
