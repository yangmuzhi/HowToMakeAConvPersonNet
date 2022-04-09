"""
卷人神经网络
"""
from .conv_base import ConvBase, LayerBase
from .utils import get_padding, normalize_func,\
     gen_stu, softmax, ReLu, get_similar_value_ind
import numpy as np

class ConvStu(ConvBase):

    def __init__(self) -> None:

        super().__init__()

    def build_layers(self):
        m = gen_stu(size=10, mode="uniform")
        layer_input = LayerInput(name="input", inputs=m, normalize=True)
        kernel = np.array([[ 0.1,  0.1,  0.1],
                        [ 0.1, -0.8,  0.1],
                        [ 0.1,  0.1,  0.1]])
        layer_conv = LayerConv(name="conv", inputs=layer_input,
                                kernel=kernel, stride=1)
        layer_max_pool = LayerMaxpooling(name='max_pooling',
                                    inputs=layer_conv,stride=1)
        dims = 2
        x = layer_max_pool.outputs.reshape([1,-1])
        w = np.random.normal(0,1,x.shape[1]*dims).reshape(x.shape[1],-1)
        b = np.random.normal(0,1,dims).reshape(-1,dims)
        layer_dense = LayerDense(name="dense", inputs=layer_max_pool, weights=w, 
                                    bias=b, kernel="relu")

        return layer_dense

# layer Input类
class LayerInput(LayerBase):

    def __init__(self, name, inputs, normalize=False) -> None:

        super().__init__(name, inputs, normalize)
    
    def _inner_process(self, inputs, normalize):
        """
        layer处理inputs的方法，如何卷的实现；需要override
            params: inputs (np.array)
            return: output (np.array)
        """
        if normalize:
            outputs = normalize_func(inputs)
        else:
            outputs = inputs
        return outputs

# layer Conv类
class LayerConv(LayerBase):

    def __init__(self, name, inputs, kernel=None, 
                    stride=0, mode="VALID") -> None:
        super().__init__(name, inputs, kernel, stride, mode)
    
    def _inner_process(self, inputs, kernel, stride, mode="VALID"):
        """
        layer处理Conv的方法，如何卷的实现；需要override
            params: inputs (np.array)
            params: kernel (np.array) 2-d dims
            params: stride (int)
            params: mode   (string)
            return: output (np.array)
        """
        ks = kernel.shape
        # get_padding
        pad = get_padding(ks, mode=mode)
        padded_inputs = np.pad(inputs, pad_width=((pad[0], pad[2]), (pad[1], pad[3])), mode="constant")

        height, width = inputs.shape
        out_width = int((width + pad[0] + pad[2] - ks[0]) / stride + 1)
        out_height = int((height + pad[1] + pad[3] - ks[1]) / stride + 1)

        outputs = np.empty(shape=(out_height, out_width))
        self.id_map_output = np.empty(shape=(out_height, out_width))
        # print(padded_inputs)
        for r, y in enumerate(range(0, padded_inputs.shape[0]-ks[1]+1, stride)):
            for c, x in enumerate(range(0, padded_inputs.shape[1]-ks[0]+1, stride)):
                # res 卷的结果
                res = padded_inputs[y:y+ks[1], x:x+ks[0]] * kernel
                outputs[r][c] = np.sum(res)
                outputs = ReLu(outputs)
                # 修改idmap
                ind = get_similar_value_ind(padded_inputs[y:y+ks[1], x:x+ks[0]], np.sum(res), 
                                            y, x, pad, height, width)
                # print("=====================")
                # print(padded_inputs[y:y+ks[1], x:x+ks[0]])
                # print("ind: ", ind)
                if not ind:
                    self.id_map_output[r][c] = -1
                    continue
                self.id_map_output[r][c] = self.id_map_input[ind[0]][ind[1]]
        return outputs


# layer kernel 类
class LayerMaxpooling(LayerBase):

    def __init__(self, name, inputs, kernel_size=[3,3], stride=0, mode="VALID") -> None:
        
        self.kernel_size = kernel_size

        super().__init__(name, inputs, stride, mode)
    
    def _inner_process(self, inputs, stride, mode="VALID"):
        """
        layer处理Conv的方法，如何卷的实现；需要override
            params: inputs (np.array)
            params: stride (int)
            params: mode   (string)
            return: output (np.array)
        """
        ks = self.kernel_size
        # get_padding
        pad = get_padding(ks, mode=mode)
        padded_inputs = np.pad(inputs, pad_width=((pad[0], pad[2]), (pad[1], pad[3])), mode="constant")

        height, width = inputs.shape
        out_width = int((width + pad[0] + pad[2] - ks[0]) / stride + 1)
        out_height = int((height + pad[1] + pad[3] - ks[1]) / stride + 1)

        outputs = np.empty(shape=(out_height, out_width))
        self.id_map_output = np.empty(shape=(out_height, out_width))
        # print(padded_inputs)
        for r, y in enumerate(range(0, padded_inputs.shape[0]-ks[1]+1, stride)):
            for c, x in enumerate(range(0, padded_inputs.shape[1]-ks[0]+1, stride)):
                # res 卷的结果
                res = padded_inputs[y:y+ks[1], x:x+ks[0]]
                outputs[r][c] = np.max(res)
                # 修改idmap
                ind = get_similar_value_ind(padded_inputs[y:y+ks[1], x:x+ks[0]], np.max(res), 
                                            y, x, pad, height, width)
                # print("=====================")
                # print(padded_inputs[y:y+ks[1], x:x+ks[0]])
                # print("ind: ", ind)
                if not ind:
                    self.id_map_output[r][c] = -1
                    continue
                self.id_map_output[r][c] = self.id_map_input[ind[0]][ind[1]]
        return outputs

class LayerDense(LayerBase):
    """
    全连接，dense，绝胜出最终的卷王
    """
    def __init__(self, name, inputs, *args, **kwargs) -> None:
        
        super().__init__(name, inputs, *args, **kwargs)
        
    
    def _inner_process(self, inputs, weights, bias, kernel="softmax"):
        x = inputs.reshape([1,-1])
        w = weights.reshape(x.shape[1],-1)
        outputs = np.dot(x,w) + bias

        self.id_map_input = self.id_map_input.reshape(-1)

        if kernel == "softmax":
            outputs = softmax(outputs)
            self.id_map_output = self.id_map_input[np.argmax(outputs)]
            print("argmax ", np.argmax(outputs))
            print("outputs ", outputs)
            print("id_map_input ", self.id_map_input)
        elif kernel == "None":
            self.id_map_output = self.id_map_input
            return outputs
        elif kernel == "relu":
            outputs = ReLu(outputs)
            self.id_map_output = self.id_map_input[np.argmax(outputs)]
            if self.id_map_input[np.argmax(outputs)] <= 0:
                self.id_map_output = -1 
        return outputs