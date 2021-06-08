import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer_conv2d, xavier_initializer


def convolution(x, weight_shape, stride, padding="SAME", bias=True, initializer=xavier_initializer_conv2d(seed=0)):
    weight = tf.Variable(initializer(shape=weight_shape))
    x = tf.nn.conv2d(x, weight, strides=stride, padding=padding)
    if bias:
        return tf.add(x, tf.Variable(tf.zeros([weight_shape[-1]]), dtype=tf.float32))
    else:
        return x


def full_connected(x, weight_shape, bias=True, initializer=xavier_initializer(seed=0)):
    weight = tf.Variable(initializer(shape=weight_shape))
    x = tf.matmul(x, weight)
    if bias:
        return tf.add(x, tf.Variable(tf.zeros([weight_shape[-1]]), dtype=tf.float32))
    else:
        return x


class LSTM(object):
    def __init__(self, network_architecture, learning_rate=0.0001, load_model=None, gradient_clip=None, batch_norm=None, keep_prob=1.0):

        self.network_architecture = network_architecture
        self.binary_class = True if self.network_architecture["label_size"] == 2 else False
        self.learning_rate = learning_rate
        self.gradient_clip = gradient_clip
        self.batch_norm = batch_norm
        self.keep_prob = keep_prob

        self._create_network()
        tf.summary.scalar("loss", self.loss)
        tf.summary.scalar("accuracy", self.accuracy)
        self.sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
        self.summary = tf.summary.merge_all()
        if load_model:
            tf.reset_default_graph()
            self.saver.restore(self.sess, load_model)

    def _create_network(self):
        self.x = tf.placeholder(tf.float32, [None] + self.network_architecture["input_word"], name="input_word")
        if self.binary_class:
            self.y = tf.placeholder(tf.float32, [None], name="output")
        else:
            self.y = tf.placeholder(tf.float32, [None, self.network_architecture["label_size"]], name="output")
        self.is_train = tf.placeholder_with_default(False, [])
        keep_prob = tf.where(self.is_train, self.keep_prob, 1.0)
        _layer_norm = False

        with tf.variable_scope("word_level"):
            cell_bw, cell_fw = [], []
            for i in range(1, 4):
                cell = tf.contrib.rnn.LayerNormBasicLSTMCell(num_units=self.network_architecture["n_hidden_%i" % i], dropout_keep_prob=keep_prob,layer_norm=_layer_norm)
                cell_fw.append(cell)
                cell = tf.contrib.rnn.LayerNormBasicLSTMCell(num_units=self.network_architecture["n_hidden_%i" % i], dropout_keep_prob=keep_prob,layer_norm=_layer_norm)
                cell_bw.append(cell)
            cell_bw, cell_fw = tf.contrib.rnn.MultiRNNCell(cell_bw), tf.contrib.rnn.MultiRNNCell(cell_fw)

            _layer = tf.nn.dropout(self.x, keep_prob)
            (output_fw, output_bw), (states_fw, states_bw) = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell_fw, cell_bw=cell_bw, dtype=tf.float32, inputs=_layer)
            cell_word = tf.concat([states_fw[-1][-1], states_bw[-1][-1]], axis=1)

        _layer = tf.nn.dropout(cell_word, keep_prob)
        _shape = _layer.shape.as_list()

        if self.binary_class:
            _weight = [_shape[-1], 1]

            _layer = tf.nn.dropout(_layer, keep_prob)
            _layer = full_connected(_layer, _weight)

            if self.batch_norm is not None:
                _layer = tf.contrib.layers.batch_norm(_layer, decay=self.batch_norm, is_training=self.is_train, updates_collections=None)

            self.prediction = tf.sigmoid(tf.squeeze(_layer, axis=1))
            _loss = self.y * tf.log(self.prediction + 1e-8) + (1 - self.y) * tf.log(1 - self.prediction + 1e-8)
            self.loss = - tf.reduce_mean(_loss)
            _prediction = tf.cast((self.prediction > 0.5), tf.float32)
            self.accuracy = 1 - tf.reduce_mean(tf.abs(self.y - _prediction))
        else:
            _weight = [_shape[-1], self.network_architecture["label_size"]]

            _layer = tf.nn.dropout(_layer, keep_prob)
            _layer = full_connected(_layer, _weight)

            if self.batch_norm is not None:
                _layer = tf.contrib.layers.batch_norm(_layer, decay=self.batch_norm, is_training=self.is_train, updates_collections=None)

            self.prediction = tf.nn.softmax(_layer)
            self.loss = - tf.reduce_sum(self.y * tf.log(self.prediction + 1e-8))
            correct_prediction = tf.equal(tf.argmax(self.y, 1), tf.argmax(self.prediction, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        self.lr_decay = tf.placeholder_with_default(1.0, [])
        optimizer = tf.train.AdamOptimizer(self.learning_rate / self.lr_decay)
        if self.gradient_clip is not None:
            var = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, var), self.gradient_clip)
            self.train = optimizer.apply_gradients(zip(grads, var))
        else:
            self.train = optimizer.minimize(self.loss)

        self.saver = tf.train.Saver()


if __name__ == '__main__':
    import os

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    net = {
        "input_word": [40, 300], "label_size": 2,
        "n_hidden_1": 64, "n_hidden_2": 128, "n_hidden_3": 256,
    }
    LSTM(net)
