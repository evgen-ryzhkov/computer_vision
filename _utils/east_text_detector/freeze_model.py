# getting frozen (.pb) model for production (opencv using)
# from default tensorflow model

# base theory and examples about freezing
#  - more info https://cv-tricks.com/how-to/freeze-tensorflow-models/
#  - https://stackoverflow.com/questions/45864363/tensorflow-how-to-convert-meta-data-and-index-model-files-into-one-graph-pb

# final code was made according to comments https://github.com/opencv/opencv/issues/12491
# also it needs to change code in model.py according to https://github.com/opencv/opencv/issues/12491#issuecomment-420351432

# -------------------------------------------------------

# how to use
# freeze model:
#   python freeze_model.py

import os
import tensorflow as tf
import model


def freeze_graph():
    model_dir = 'tmp/'
    frozen_model_path = 'freeze_models/frozen_custom_east_text_detection.pb'
    # choose which outputs you need
    # for text detection uses these two outputs
    output_node_names = ["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"]

    # don't know how it can do without these lines of code from eval.py
    input_images = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='input_images')
    global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
    f_score, f_geometry = model.model(input_images, is_training=False)
    variable_averages = tf.train.ExponentialMovingAverage(0.997, global_step)

    saver = tf.train.Saver(variable_averages.variables_to_restore())

    ckpt_state = tf.train.get_checkpoint_state(model_dir)
    model_path = os.path.join(model_dir, os.path.basename(ckpt_state.model_checkpoint_path))

    with tf.Session() as sess:
        saver.restore(sess, model_path)

        # Freeze the graph
        output_graph_def = tf.graph_util.convert_variables_to_constants(
            sess,
            tf.get_default_graph().as_graph_def(),
            output_node_names)

        # Finally we serialize and dump the output graph to the filesystem
        with tf.gfile.GFile(frozen_model_path, "wb") as f:
            f.write(output_graph_def.SerializeToString())
        print("%d ops in the final graph." % len(output_graph_def.node))
        print('[OK] The frozen model have been successfully saved.')


if __name__ == '__main__':
    freeze_graph()
