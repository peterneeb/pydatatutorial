from keras.layers.wrappers import Wrapper  # pylint: disable=g-import-not-at-top
from keras.models import Sequential  # pylint: disable=g-import-not-at-top
import numpy as np


def getOutputDims(layer):
    try:
        outputlabels = ppDimTuple(layer.output_shape)
    except AttributeError:
        outputlabels = 'multiple'
    return outputlabels

def getInputDims(layer):
    if isinstance(layer.input_shape, tuple):
        inputlabels = ppDimTuple(layer.input_shape)
    elif isinstance(layer.input_shape, list):
        inputlabels = ', '.join([ppDimTuple(ishape) for ishape in layer.input_shape])
    else:
        inputlabels = 'multiple'
    return inputlabels

def getNoChannels(layer):
    return layer.input_shape[3]

def getLayerName(layer):
    layer_name = layer.name
    class_name = layer.__class__.__name__

    if isinstance(layer, Wrapper):
        layer_name = '{}({})'.format(layer_name, layer.layer.name)
        child_class_name = layer.layer.__class__.__name__
        class_name = 'wrapper {}({})'.format(class_name, child_class_name)

    return  '{}\n\<{}\>'.format(layer_name, class_name)

def getActivation(layer):
    official_activations = ['softmax','relu','elu','selu','softplus','softsign','sigmoid','tanh','linear','hard_sigmoid',
                            'leakyrelu','prelu','thresholdedrelu']
    if layer.output._op.type:
        act = str(layer.output._op.type)
        if act.lower() not in official_activations:
            act = '-'
        return act
    else:
        return None

def getPadding(input_size, kernel_size, padding_desc, stride):
    padding = None
    kernel_size = np.array(kernel_size)
    input_size = np.array(input_size[1:3])
    stride = np.array(stride)
    if padding_desc == 'same':
        output_size = input_size
        padding = "same"#((input_size-1) * stride +  kernel_size - input_size) * 0.5
        padding = (output_size-1) * stride - input_size+kernel_size
    elif padding_desc == 'valid':
        for padding_i in range(10):
            output_size = (input_size-kernel_size + 2 * padding_i)/stride+1
            for i in range(len(output_size)):
                if output_size[i] != int(output_size[i]):
                    continue
            padding = [padding_i,padding_i]
            break
    else:
        pass
    return padding


def ppDimTuple(tuple):
    result = '['
    for num, dim in enumerate(tuple):
        if dim == None:
            dim = 'n'
        result += str(dim)
        if num < len(tuple)-1:
            result += ', '
    result += ']'
    return result

class BaseVis():
    def __init__(self, layer):
        self.layer = layer

    def create_default_label(self, show_activation = False):
        name = getLayerName(self.layer)
        inputlabels = getInputDims(self.layer)
        outputlabels = getOutputDims(self.layer)
        l1 = '%s\n|' % (name)
        if show_activation:
            activation = getActivation(self.layer)
            l2 = '|{input:|output:|activation:}|{{%s}|{%s}|{%s}}' % (inputlabels,outputlabels, activation)
        else:
            l2 = '|{input:|output:}|{{%s}|{%s}}' % (inputlabels, outputlabels)
        return l1+'%s'+l2

    def create_complex_label(self,parameter, hyperparameter, show_activation = False):
        label = self.create_default_label(show_activation)
        label = label % ('{%s | %s}')
        label = label % (parameter, hyperparameter)
        return label

class DefaultVis(BaseVis):
    def create_label(self):
        label = self.create_default_label()
        label = label % ('-')
        return label

class InputLayerVis(BaseVis):
    def create_label(self):
        outputlabels = getOutputDims(self.layer)
        label = '{Input Layer | Output: %s}' % (outputlabels)
        return label

class ActivationVis(BaseVis):
    def create_label(self):
        label = '{Activation: %s}' % (getActivation(self.layer))
        return label

class AdvancedActivationVis(BaseVis):
    def create_label(self):
        if hasattr(self.layer,'alpha'):
            param = str(self.layer.alpha)
        elif hasattr(self.layer, 'theta'):
            param = str(self.layer.theta)
        else:
            param = '?'

        label = '{Activation: %s, param: %s}' % (str(self.layer.__class__.__name__),param)
        return label


class DenseVis(BaseVis):
    def create_label(self):
        no_params = self.layer.kernel.shape[0] * self.layer.kernel.shape[1] + self.layer.bias.shape[0]
        initializer = self.layer.kernel_initializer.__class__.__name__
        parameter = 'kernel: {}\nbias: {}\nno params: {}'.format(ppDimTuple(self.layer.kernel.shape),ppDimTuple(self.layer.bias.shape),no_params)
        hyperparameter = 'units: {}\nkernel init: {}'.format(self.layer.units,initializer)
        label = self.create_complex_label(parameter,hyperparameter, show_activation=True)
        return label

class Conv2DVis(BaseVis):
    def create_label(self):
        initializer = self.layer.kernel_initializer.__class__.__name__
        no_filters = self.layer.filters
        no_channels = getNoChannels(self.layer)
        padding_str = self.layer.padding
        strides = self.layer.strides
        padding = getPadding(self.layer.input_shape, self.layer.kernel_size, padding_str, strides)
        no_params = (self.layer.kernel.shape[0] * self.layer.kernel.shape[1] * no_channels +1) * no_filters
        kernel_dim = (self.layer.kernel.shape[0],self.layer.kernel.shape[1],no_channels)
        parameter = '{} kernel dim: {}\nbias: {}\nno params: {}'.format(no_filters,
                                                             ppDimTuple(kernel_dim),ppDimTuple(self.layer.bias.shape),no_params  )
        hyperparameter = 'padding: {} ({})\nstride: {}\nkernel init: {}'.format(padding, padding_str, ppDimTuple(strides),initializer)
        label = self.create_complex_label(parameter, hyperparameter, show_activation=True)
        return label

class Conv1DVis(BaseVis):
    def create_label(self):
        initializer = self.layer.kernel_initializer.__class__.__name__
        no_filters = self.layer.filters
        #no_channels = getNoChannels(self.layer)
        padding_str = self.layer.padding
        strides = self.layer.strides
        padding = getPadding(self.layer.input_shape, self.layer.kernel_size, padding_str, strides)
        no_params = (self.layer.kernel.shape[0] * self.layer.kernel.shape[1] +1) * no_filters
        kernel_dim = (self.layer.kernel.shape[0], self.layer.kernel.shape[1])

        parameter = '{} kernel dim: {}\nbias: {}\nno params: {}'.format(no_filters,
                                                             ppDimTuple(kernel_dim),ppDimTuple(self.layer.bias.shape),no_params  )
        hyperparameter = 'padding: {} ({})\nstride: {}\nkernel init: {}'.format(padding, padding_str, ppDimTuple(strides),initializer)
        label = self.create_complex_label(parameter, hyperparameter, show_activation=True)
        return label

class MaxPooling2DVis(BaseVis):
    def create_label(self):
        padding_str = self.layer.padding
        strides = self.layer.strides
        padding = getPadding(self.layer.input_shape, self.layer.pool_size, padding_str, strides)
        parameter = 'pool dim: {}\nno params: 0'.format(ppDimTuple(self.layer.pool_size))
        hyperparameter = 'padding: {} ({})\nstride: {}'.format(padding, padding_str, ppDimTuple(strides))
        label = self.create_complex_label(parameter, hyperparameter)
        return label


class DropoutVis(BaseVis):
    def create_label(self):
        parameter = 'no params: 0'
        if self.layer.noise_shape:
            noise_shape = ppDimTuple(self.layer.noise_shape)
        else:
            noise_shape = 'None'
        hyperparameter = 'drop rate: {}\nnoise shape: {}'.format(self.layer.rate, noise_shape)
        label = self.create_complex_label(parameter, hyperparameter)
        return label

class LSTMVis(BaseVis):
    def create_label(self):
        no_params = self.layer.recurrent_kernel.shape[0]*self.layer.recurrent_kernel.shape[1]\
                    + self.layer.kernel.shape[0]*self.layer.kernel.shape[1]\
                    + self.layer.bias.shape[0]
        initializer = self.layer.kernel_initializer.__class__.__name__
        parameter = 'kernel (4 gates): {}\nrecurrent kernel: {}\nbias: {}\nno params: {}'\
                    .format(ppDimTuple(self.layer.kernel.shape),ppDimTuple(self.layer.recurrent_kernel.shape),
                            ppDimTuple(self.layer.bias.shape), no_params)
        hyperparameter = 'units: {}\nkernel init: {}'.format(self.layer.units,initializer)
        label = self.create_complex_label(parameter, hyperparameter, show_activation=True)

        return label

class SimpleRNNvis(BaseVis):
    def create_label(self):
        no_params = self.layer.recurrent_kernel.shape[0]*self.layer.recurrent_kernel.shape[1]\
                    + self.layer.kernel.shape[0]*self.layer.kernel.shape[1]\
                    + self.layer.bias.shape[0]
        initializer = self.layer.kernel_initializer.__class__.__name__
        parameter = 'kernel (4 gates): {}\nrecurrent kernel: {}\nbias: {}\nno params: {}'\
                    .format(ppDimTuple(self.layer.kernel.shape),ppDimTuple(self.layer.recurrent_kernel.shape),
                            ppDimTuple(self.layer.bias.shape), no_params)
        hyperparameter = 'units: {}\nkernel init: {}'.format(self.layer.units,initializer)
        label = self.create_complex_label(parameter, hyperparameter, show_activation=True)

        return label
class GRUvis(BaseVis):
    def create_label(self):
        no_params = self.layer.recurrent_kernel.shape[0]*self.layer.recurrent_kernel.shape[1]\
                    + self.layer.kernel.shape[0]*self.layer.kernel.shape[1]\
                    + self.layer.bias.shape[0]
        initializer = self.layer.kernel_initializer.__class__.__name__
        parameter = 'kernel (3 gates): {}\nrecurrent kernel: {}\nbias: {}\nno params: {}'\
                    .format(ppDimTuple(self.layer.kernel.shape),ppDimTuple(self.layer.recurrent_kernel.shape),
                            ppDimTuple(self.layer.bias.shape), no_params)
        hyperparameter = 'units: {}\nkernel init: {}'.format(self.layer.units,initializer)
        label = self.create_complex_label(parameter, hyperparameter, show_activation=True)

        return label