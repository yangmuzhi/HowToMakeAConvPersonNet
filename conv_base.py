"""
卷人神经网络
2维矩阵，矩阵中每个数值表征一个人
"""

import numpy as np
from abc import ABC, abstractmethod


class ConvBase(ABC):
    """
    卷王生成器
    """

    def __init__(self,size=5) -> None:
        self.size = size
        self.final_layer = None
        self.final_layer = self.build_layers()
        self.all_layers = self.get_all_layers()

    def get_all_layers(self, output=False) -> list:
        """
        获得所有的layers
            return: all layers (list)
        """
        # 从final_layer 遍历
        # if not self.final_layer:
        #     return []
        res = []
        cur_layer = self.final_layer
        while cur_layer:
            # print("cur_layer: ", cur_layer)
            res.insert(0, self.get_one_layer(cur_layer, output))
            cur_layer = cur_layer.parent_layers
        return res

    def get_one_layer(self, layer, output=False) -> dict:
        name = layer.name
        value = layer.outputs
        if output:
            return {"name": name, "layer": layer, "output": value}
        return {"name": name, "layer": layer}
    
    def get_map(self):
        """get id map"""
        res = {}
        for layer_map in self.all_layers:
            name = layer_map['name']
            layer = layer_map['layer']
            res[name] = layer.id_map_output
        return res
    
    def plot_layers(self):
        """绘制layers的关系"""
        return

    @abstractmethod
    def build_layers(inputs, kernel, stride, mode="SAME"):
        return []


# layer base类
class LayerBase(ABC):

    def __init__(self, name, inputs, *args, **kwargs) -> None: 
        self.name = name
        if isinstance(inputs, LayerBase):
            # parent layers
            self.parent_layers = inputs
            inputs = inputs.outputs

            # id赋值
            self.id_map_input = self.parent_layers.id_map_output
        elif inputs is not None:
            # inputs is a value means inputlayer with no parent layers
            self.parent_layers = None
            ## init id map,用来储存和识别每一步谁卷赢了
            shape = np.array(inputs).shape
            # input 2-d dims
            if len(shape) == 2:
                # id从0开始，-1表示这个位置的所有人都已经gg了
                self.id_map_input = np.arange(shape[0]*shape[1]).reshape(shape)
                # input layer的id map不变
                self.id_map_output = self.id_map_input
            else:
                raise NotImplementedError("only 2d is supported, shape : {} is not implemented".format(shape))
        self.inputs = np.array(inputs)
        assert len(self.inputs.shape) == 2, "inputs only 2-dims is supported, this inputs shape is {}".format(self.inputs.shape)
        # process inputs
        self.outputs = self._inner_process(self.inputs, *args, **kwargs)

    @abstractmethod
    def _inner_process(self, inputs, *args, **kwargs):
        """
        layer处理inputs的方法，如何卷的实现；需要override
        """
        
        return 
    
    @property
    def shape(self):
        return self.inputs.shape