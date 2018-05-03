# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Utilities related to model visualization."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys


try:
    # pydot-ng is a fork of pydot that is better maintained.
    import pydot_ng as pydot  # pylint: disable=g-import-not-at-top
except ImportError:
    # Fall back on pydot if necessary.
    # Silence a `print` statement that occurs in case of import error,
    # by temporarily replacing sys.stdout.
    _stdout = sys.stdout
    sys.stdout = sys.stderr
    try:
        import pydot  # pylint: disable=g-import-not-at-top
    except ImportError:
        pydot = None
    finally:
        # Restore sys.stdout.
        sys.stdout = _stdout


def _check_pydot():
    try:
        # Attempt to create an image of a blank graph
        # to check the pydot/graphviz installation.
        pydot.Dot.create(pydot.Dot())
    except Exception:
        # pydot raises a generic Exception here,
        # so no specific class can be caught.
        raise ImportError('Failed to import pydot. You must install pydot'
                          ' and graphviz for `pydotprint` to work.')


from keras.layers.wrappers import Wrapper  # pylint: disable=g-import-not-at-top
from keras.models import Sequential  # pylint: disable=g-import-not-at-top
from vis.vis_classes2 import DefaultVis, DenseVis, Conv2DVis, InputLayerVis, MaxPooling2DVis, DropoutVis, LSTMVis
from vis.vis_classes2 import ActivationVis, AdvancedActivationVis, Conv1DVis, SimpleRNNvis, GRUvis

def getDefaultVisDict():
    visdict = {}
    visdict["Dense"] = DenseVis
    visdict["InputLayer"] = InputLayerVis
    visdict["MaxPooling2D"] = MaxPooling2DVis
    visdict["Dropout"] = DropoutVis
    visdict["Conv2D"] = Conv2DVis
    visdict["Conv1D"] = Conv1DVis
    visdict["Activation"] = ActivationVis
    visdict["LeakyReLU"] = AdvancedActivationVis
    visdict["ELU"] = AdvancedActivationVis
    visdict["ThresholdedReLU"] =AdvancedActivationVis
    visdict["LSTM"] =LSTMVis
    visdict["SimpleRNN"] =SimpleRNNvis
    visdict["GRU"] = GRUvis

    return visdict


def model_to_dot2(model, rankdir, nondefaultvisdict = None):

    _check_pydot()
    dot = pydot.Dot()
    dot.set('concentrate', True)
    dot.set('rankdir',rankdir)
    dot.set_node_defaults(shape='Mrecord')

    if isinstance(model, Sequential):
        if not model.built:
            model.build()
        model = model.model
    layers = model.layers

    # Add Visualisation dictionaire

    visdict = getDefaultVisDict()
    if nondefaultvisdict:
        visdict.update(nondefaultvisdict)

    # Create graph nodes.
    for layer in layers:
        class_name = layer.__class__.__name__
        if class_name in visdict.keys():
            Visualisation = visdict[class_name]
        else:
            Visualisation = DefaultVis
        label = Visualisation(layer).create_label()

        node = pydot.Node(str(id(layer)), label=label)
        dot.add_node(node)

    # Connect nodes with edges.
    for layer in layers:
        layer_id = str(id(layer))
        for i, node in enumerate(layer.inbound_nodes):
            node_key = layer.name + '_ib-' + str(i)
            if node_key in model.container_nodes:
                for inbound_layer in node.inbound_layers:
                    inbound_layer_id = str(id(inbound_layer))
                    layer_id = str(id(layer))
                    dot.add_edge(pydot.Edge(inbound_layer_id, layer_id))
    return dot


def plot_model2(model,
                to_file='model.png',
                rankdir='TB',
                nondefaultvisdict = None):
    """Converts a Keras model to dot format and save to a file.
  
    Arguments:
        model: A Keras model instance
        to_file: File name of the plot image.
        rankdir: `rankdir` argument passed to PyDot,
            a string specifying the format of the plot:
            'TB' creates a vertical plot;
            'LR' creates a horizontal plot.
    """
    dot = model_to_dot2(model, rankdir, nondefaultvisdict)
    _, extension = os.path.splitext(to_file)
    if not extension:
        extension = 'png'
    else:
        extension = extension[1:]
    dot.write(to_file, format=extension)
