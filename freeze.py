import tensorflow as tf
from tensorflow.python.tools import freeze_graph

freeze_graph.freeze_graph(input_graph='./graph.pb',
                          input_saver="",
                          input_binary=True,
                          input_checkpoint='./models/model.ckpt',
                          output_node_names='actions',
                          restore_op_name="",
                          filename_tensor_name="",
                          output_graph='./frozen_actions.pb',
                          clear_devices=False,
                          initializer_nodes="")
