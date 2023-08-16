# !/usr/bin/env python
# coding: utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from alps.common.model import BaseModel
import tfplus
import keras
from tensorflow.contrib.layers.python.layers.embedding_ops import embedding_lookup_unique


class BARL(BaseModel):

    def __init__(self, config):
        super(BARL, self).__init__(config)
        self.y = config.y[0].feature_name
        self._config = self.config
        self._customer_outputs = {}
        self.hidden_dim = self.config.model_def.embedding_size

    def build_model(self, inputs, labels):
        ## config
        tf.set_random_seed(1234)

        ## input
        item_title_fea = inputs['item_title_fea']
        item_title_ctn = inputs['item_title_ctn']

        item_keyword_fea = inputs['item_keyword_fea']
        item_keyword_ctn = inputs['item_keyword_ctn']

        query_fea = inputs['query_fea']
        query_ctn = inputs['query_ctn']

        item_title_seq = tf.reshape(inputs['item_title_seq'], [-1, 20, 16])
        item_title_ctn_seq = inputs['item_title_ctn_seq']
        item_keyword_seq = tf.reshape(inputs['item_keyword_seq'], [-1, 20, 48])
        item_keyword_ctn_seq = inputs['item_keyword_ctn_seq']
        item_len = inputs['item_len']

        query_fea_seq = tf.reshape(inputs['query_fea_seq'], [-1, 20, 16])
        query_ctn_seq = inputs['query_ctn_seq']
        query_len = inputs['query_len']

        word_lookup_table = tfplus.get_kv_variable("test_var_name",
                                                   embedding_dim=self.hidden_dim,
                                                   key_dtype=tf.string,
                                                   value_dtype=tf.float32,
                                                   initializer=tf.keras.initializers.he_normal(),
                                                   partitioner=tf.fixed_size_partitioner(
                                                       num_shards=self.config.model_def.num_shards),
                                                   trainable=True,
                                                   enter_threshold=0)

        target_title_embedding = embedding_lookup_unique(word_lookup_table, item_title_fea)  # B*16*d
        target_title_embedding = tf.layers.dropout(target_title_embedding, 0.3)
        target_title_embedding = tf.nn.l2_normalize(target_title_embedding, axis=-1)

        target_keyword_embedding = embedding_lookup_unique(word_lookup_table, item_keyword_fea)  # B*48*d
        target_keyword_embedding = tf.layers.dropout(target_keyword_embedding, 0.3)
        target_keyword_embedding = tf.nn.l2_normalize(target_keyword_embedding, axis=-1)

        target_title_mask = tf.sequence_mask(tf.reshape(item_title_ctn, [-1]), 16)
        target_title_embedding_whole = tf.reduce_sum(
            target_title_embedding * tf.expand_dims(tf.cast(target_title_mask, tf.float32), -1), 1)

        target_keyword_mask = tf.sequence_mask(tf.reshape(item_keyword_ctn, [-1]), 48)
        target_keyword_embedding_whole = tf.reduce_sum(
            target_keyword_embedding * tf.expand_dims(tf.cast(target_keyword_mask, tf.float32), -1), 1)

        target_item_embedding_whole = tf.nn.l2_normalize(
            tf.layers.dense(tf.concat([target_title_embedding_whole, target_keyword_embedding_whole], 1), self.hidden_dim,
                            activation=tf.nn.relu), axis=-1)

        target_query_embedding = embedding_lookup_unique(word_lookup_table, query_fea)
        target_query_embedding = tf.layers.dropout(target_query_embedding, 0.3)
        target_query_embedding = tf.nn.l2_normalize(target_query_embedding, axis=-1)

        target_query_mask = tf.sequence_mask(tf.reshape(query_ctn, [-1]), 16)
        target_query_embedding_whole = tf.nn.l2_normalize(
            tf.reduce_sum(target_query_embedding * tf.expand_dims(tf.cast(target_query_mask, tf.float32), -1), 1), axis=-1)

        item_behavior_neighbor_id_seq_mask = tf.sequence_mask(tf.reshape(item_len, [-1]), 20)

        item_behavior_neighbor_title_seq_embedding = tf.layers.dropout(embedding_lookup_unique(word_lookup_table, item_title_seq),
                                                     0.3)  # B*20*16*d
        item_behavior_neighbor_title_seq_mask = tf.sequence_mask(item_title_ctn_seq, 16)  # B*20*16
        item_behavior_neighbor_title_seq_embedding = tf.nn.l2_normalize(
            tf.reduce_sum(item_behavior_neighbor_title_seq_embedding * tf.expand_dims(tf.cast(item_behavior_neighbor_title_seq_mask, tf.float32), -1), 2),
            axis=-1)  # B*20*d

        item_behavior_neighbor_keyword_seq_embedding = tf.layers.dropout(embedding_lookup_unique(word_lookup_table, item_keyword_seq),
                                                       0.3)  # B*20*48*d
        item_behavior_neighbor_keyword_seq_mask = tf.sequence_mask(item_keyword_ctn_seq, 48)  # B*20*48
        item_behavior_neighbor_keyword_seq_embedding = tf.nn.l2_normalize(
            tf.reduce_sum(item_behavior_neighbor_keyword_seq_embedding * tf.expand_dims(tf.cast(item_behavior_neighbor_keyword_seq_mask, tf.float32), -1),
                          2), axis=-1)  # B*20*d

        item_behavior_neighbor_seq_emb = tf.layers.dense(
            tf.stack([item_behavior_neighbor_title_seq_embedding, item_behavior_neighbor_keyword_seq_embedding], axis=1),
            self.hidden_dim, activation=None)  # B*2*d
        item_behavior_neighbor_seq_gate = tf.nn.softmax(tf.layers.dense(item_behavior_neighbor_seq_emb, 1), axis=1)  # B*3*1
        item_behavior_neighbor_seq_embedding = tf.reduce_sum(item_behavior_neighbor_seq_emb * item_behavior_neighbor_seq_gate, axis=1) # B*20*d

        query_behavior_neighbor_id_seq_mask = tf.sequence_mask(tf.reshape(query_len, [-1]), 20)  # B*20

        query_behavior_neighbor_fea_seq_embedding = tf.layers.dropout(embedding_lookup_unique(word_lookup_table, query_fea_seq),
                                                    0.3)  # B*20*16*d
        query_behavior_neighbor_fea_seq_mask = tf.sequence_mask(query_ctn_seq, 16)  # B*20*16
        query_behavior_neighbor_fea_seq_embedding = tf.nn.l2_normalize(
            tf.reduce_sum(query_behavior_neighbor_fea_seq_embedding * tf.expand_dims(tf.cast(query_behavior_neighbor_fea_seq_mask, tf.float32), -1), 2),
            axis=-1)  # B*20*d

        query_behavior_neighbor_seq_embedding = query_behavior_neighbor_fea_seq_embedding

        item_t = self.embedding(
            tf.tile(tf.expand_dims(tf.range(tf.shape(item_behavior_neighbor_seq_embedding)[1]), 0),
                    [tf.shape(item_behavior_neighbor_seq_embedding)[0], 1]),
            vocab_size=20,
            num_units=self.hidden_dim,
            scope='item_pos_emb')

        query_t = self.embedding(
            tf.tile(tf.expand_dims(tf.range(tf.shape(query_behavior_neighbor_seq_embedding)[1]), 0),
                    [tf.shape(query_behavior_neighbor_seq_embedding)[0], 1]),
            vocab_size=20,
            num_units=self.hidden_dim,
            scope='query_pos_emb')

        item_behavior_neighbor_seq_embedding += item_t
        query_behavior_neighbor_seq_embedding += query_t

        query_behavior_neighbor_seq_embedding = tf.layers.dropout(query_behavior_neighbor_seq_embedding, 0.3) * tf.expand_dims(
            tf.cast(query_behavior_neighbor_id_seq_mask, tf.float32), -1)
        item_behavior_neighbor_seq_embedding = tf.layers.dropout(item_behavior_neighbor_seq_embedding, 0.3) * tf.expand_dims(
            tf.cast(item_behavior_neighbor_id_seq_mask, tf.float32), -1)

        # Self-attention
        query_behavior_neighbor_seq_embedding = self.multihead_attention(queries=self.normalize(query_behavior_neighbor_seq_embedding),
                                                           keys=query_behavior_neighbor_seq_embedding,
                                                           num_units=self.hidden_dim,
                                                           num_heads=4,
                                                           dropout_rate=0.2,
                                                           causality=False,
                                                           scope="self_attention",
                                                           reuse=tf.AUTO_REUSE)

        # Feed forward
        query_behavior_neighbor_seq_embedding = self.feedforward(self.normalize(query_behavior_neighbor_seq_embedding),
                                                   num_units=[self.hidden_dim, self.hidden_dim],
                                                   dropout_rate=0.2, scope='query_feed')
        query_behavior_neighbor_seq_embedding *= tf.expand_dims(tf.cast(query_behavior_neighbor_id_seq_mask, tf.float32), -1)

        query_behavior_neighbor_seq_embedding_whole = tf.nn.l2_normalize(tf.reduce_sum(self.normalize(query_behavior_neighbor_seq_embedding), 1),axis=-1)

        # Self-attention
        item_behavior_neighbor_seq_embedding = self.multihead_attention(queries=self.normalize(item_behavior_neighbor_seq_embedding),
                                                            keys=item_behavior_neighbor_seq_embedding,
                                                            num_units=self.hidden_dim,
                                                            num_heads=4,
                                                            dropout_rate=0.2,
                                                            causality=False,
                                                            scope="self_attention",
                                                            reuse=tf.AUTO_REUSE)

        # Feed forward
        item_behavior_neighbor_seq_embedding = self.feedforward(self.normalize(item_behavior_neighbor_seq_embedding),
                                                    num_units=[self.hidden_dim, self.hidden_dim],
                                                    dropout_rate=0.2, scope='item_feed')
        item_behavior_neighbor_seq_embedding *= tf.expand_dims(tf.cast(item_behavior_neighbor_id_seq_mask, tf.float32), -1)

        item_behavior_neighbor_seq_embedding_whole = tf.nn.l2_normalize(tf.reduce_sum(self.normalize(item_behavior_neighbor_seq_embedding), 1),
                                                            axis=-1)

        query_behavior_neighbor_seq_embedding = self.target_attention(query=item_behavior_neighbor_seq_embedding_whole,
                                                        values=query_behavior_neighbor_seq_embedding,
                                                        lens=item_len,
                                                        num_units=128,
                                                        name='query_behavior_neighbor_att')

        item_behavior_neighbor_seq_embedding = self.target_attention(query=query_behavior_neighbor_seq_embedding_whole,
                                                         values=item_behavior_neighbor_seq_embedding,
                                                         lens=query_len,
                                                         num_units=128,
                                                         name='item_behavior_neighbor_att')

        item_aware_query_embedding = self.target_attention(query=target_item_embedding_whole,
                                                          values=target_query_embedding,
                                                          lens=query_ctn,
                                                          num_units=128,
                                                          name='target_item_aware_att')

        query_aware_title_embedding = self.target_attention(query=target_query_embedding_whole,
                                                                values=target_title_embedding,
                                                                lens=item_title_ctn,
                                                                num_units=128,
                                                                name='target_query_aware_title_att')

        query_aware_keyword_embedding = self.target_attention(query=target_query_embedding_whole,
                                                                  values=target_keyword_embedding,
                                                                  lens=item_keyword_ctn,
                                                                  num_units=128,
                                                                  name='target_query_aware_keyword_att')

        text_emb = tf.concat([target_item_embedding_whole, target_query_embedding_whole, item_aware_query_embedding, query_aware_title_embedding, query_aware_keyword_embedding], axis=-1)

        text_embedding = tf.layers.dense(text_emb,
                                         self.hidden_dim * 2,
                                         kernel_initializer=tf.keras.initializers.he_normal(),
                                         activation=tf.nn.relu)

        text_embedding = tf.layers.dense(text_embedding,
                                         self.hidden_dim,
                                         kernel_initializer=tf.keras.initializers.he_normal(),
                                         activation=tf.nn.relu)

        text_logits = tf.layers.dense(text_embedding, 1, kernel_initializer=tf.keras.initializers.he_normal())

        behavior_emb = tf.concat([query_behavior_neighbor_seq_embedding_whole, item_behavior_neighbor_seq_embedding_whole, query_behavior_neighbor_seq_embedding, item_behavior_neighbor_seq_embedding], axis=-1)

        seq_embedding = tf.layers.dense(behavior_emb,
                                        self.hidden_dim * 2,
                                        kernel_initializer=tf.keras.initializers.he_normal(),
                                        activation=tf.nn.relu)
        seq_embedding = tf.layers.dense(seq_embedding,
                                        self.hidden_dim,
                                        kernel_initializer=tf.keras.initializers.he_normal(),
                                        activation=tf.nn.relu)

        seq_logits = tf.layers.dense(seq_embedding, 1, kernel_initializer=tf.keras.initializers.he_normal())

        gate = tf.layers.dense(tf.concat([text_emb,behavior_emb],axis=-1), 1, activation=tf.nn.sigmoid)
        logits = text_logits*gate+(1-gate)*seq_logits

        y = tf.cast(labels[self.y], tf.float32)

        logits = tf.reshape(logits, [-1, 1])

        y = tf.reshape(y, [-1, 1])

        self.loss = tf.losses.sigmoid_cross_entropy(y, tf.cast(logits, tf.float32)) + 0.2 * tf.reduce_mean(
            tf.square(tf.cast(tf.nn.sigmoid(text_logits), tf.float32) - tf.cast(tf.nn.sigmoid(seq_logits), tf.float32)))

        self.loss += self.calc_ssl_loss(item_behavior_neighbor_seq_embedding_whole, target_query_embedding_whole, 0.2, 0.1)
        self.loss += self.calc_ssl_loss(query_behavior_neighbor_seq_embedding_whole, target_item_embedding_whole, 0.2, 0.1)

        self.predict_result = tf.nn.sigmoid(logits, name='sigmoid')

        self._customer_outputs.update(
            {"prediction": logits})

        tf.summary.scalar('loss', self.loss)

        self.customer_result = None
        return self.predict_result

    def get_prediction_result(self, **options):
        return self.predict_result

    def get_loss(self, **options):
        return self.loss

    def get_summary_op(self):
        for var in tf.trainable_variables():
            tf.summary.histogram(var.name, var)
        return tf.summary.merge_all(), None

    def customer_outputs(self):
        return self._customer_outputs

    def get_metrics(self):
        result = {'loss': self.loss}
        return result

    def embedding(self,
                  inputs,
                  vocab_size,
                  num_units,
                  scope="embedding",
                  reuse=None):

        with tf.variable_scope(scope, reuse=reuse):
            lookup_table = tf.get_variable('lookup_table',
                                           dtype=tf.float32,
                                           shape=[vocab_size, num_units])

            outputs = tf.nn.embedding_lookup(lookup_table, inputs)

        return outputs

    def multihead_attention(self, queries,
                            keys,
                            num_units=None,
                            num_heads=8,
                            dropout_rate=0,
                            causality=False,
                            scope="multihead_attention",
                            reuse=None,
                            with_qk=False):

        with tf.variable_scope(scope, reuse=reuse):
            # Set the fall back option for num_units
            if num_units is None:
                num_units = queries.get_shape().as_list[-1]

            Q = tf.layers.dense(queries, num_units, activation=None)  # (N, T_q, C)
            K = tf.layers.dense(keys, num_units, activation=None)  # (N, T_k, C)
            V = tf.layers.dense(keys, num_units, activation=None)  # (N, T_k, C)

            # Split and concat
            Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)  # (h*N, T_q, C/h)
            K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)
            V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)

            # Multiplication
            outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))  # (h*N, T_q, T_k)

            # Scale
            outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)

            # Key Masking
            key_masks = tf.sign(tf.reduce_sum(tf.abs(keys), axis=-1))  # (N, T_k)
            key_masks = tf.tile(key_masks, [num_heads, 1])  # (h*N, T_k)
            key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(queries)[1], 1])  # (h*N, T_q, T_k)

            paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
            outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs)  # (h*N, T_q, T_k)

            # Causality = Future blinding
            if causality:
                diag_vals = tf.ones_like(outputs[0, :, :])  # (T_q, T_k)
                tril = tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense()  # (T_q, T_k)
                masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(outputs)[0], 1, 1])  # (h*N, T_q, T_k)

                paddings = tf.ones_like(masks) * (-2 ** 32 + 1)
                outputs = tf.where(tf.equal(masks, 0), paddings, outputs)  # (h*N, T_q, T_k)

            # Activation
            outputs = tf.nn.softmax(outputs)  # (h*N, T_q, T_k)

            # Query Masking
            query_masks = tf.sign(tf.reduce_sum(tf.abs(queries), axis=-1))  # (N, T_q)
            query_masks = tf.tile(query_masks, [num_heads, 1])  # (h*N, T_q)
            query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(keys)[1]])  # (h*N, T_q, T_k)
            outputs *= query_masks  # broadcasting. (N, T_q, C)

            # Dropouts
            outputs = tf.layers.dropout(outputs, rate=dropout_rate)

            # Weighted sum
            outputs = tf.matmul(outputs, V_)  # ( h*N, T_q, C/h)

            # Restore shape
            outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)  # (N, T_q, C)

            # Residual connection
            outputs += queries

            # Normalize
            # outputs = normalize(outputs) # (N, T_q, C)

        if with_qk:
            return Q, K
        else:
            return outputs

    def feedforward(self, inputs,
                    num_units=[2048, 512],
                    scope="multihead_attention",
                    dropout_rate=0.2,
                    reuse=None):

        with tf.variable_scope(scope, reuse=reuse):
            # Inner layer
            params = {"inputs": inputs, "filters": num_units[0], "kernel_size": 1,
                      "activation": tf.nn.relu, "use_bias": True}
            outputs = tf.layers.conv1d(**params)
            outputs = tf.layers.dropout(outputs, rate=dropout_rate)
            # Readout layer
            params = {"inputs": outputs, "filters": num_units[1], "kernel_size": 1,
                      "activation": None, "use_bias": True}
            outputs = tf.layers.conv1d(**params)
            outputs = tf.layers.dropout(outputs, rate=dropout_rate)

            # Residual connection
            outputs += inputs

            # Normalize
            # outputs = normalize(outputs)

        return outputs

    def normalize(self, inputs,
                  epsilon=1e-8,
                  scope="ln",
                  reuse=None):

        with tf.variable_scope(scope, reuse=reuse):
            inputs_shape = inputs.get_shape()
            params_shape = inputs_shape[-1:]

            mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
            beta = tf.Variable(tf.zeros(params_shape))
            gamma = tf.Variable(tf.ones(params_shape))
            normalized = (inputs - mean) / ((variance + epsilon) ** (.5))
            outputs = gamma * normalized + beta

        return outputs

    def calc_ssl_loss(self, norm_input1, norm_input2, temp, reg):  # B*d
        '''
        Calculating SSL loss
        '''
        ttl_score = tf.matmul(norm_input1, norm_input2, transpose_a=False, transpose_b=True) / temp  # B*B

        # for numerical stability
        ttl_max = tf.reduce_max(ttl_score, axis=1, keepdims=True)
        ttl_score = ttl_score - tf.stop_gradient(ttl_max)

        a = tf.diag(norm_input1[:, 0])
        paddings = tf.ones_like(a) * 0
        other_mask = tf.where(tf.equal(a, 0), tf.ones_like(a), paddings)
        mask = tf.where(tf.equal(other_mask, 0), tf.ones_like(a), paddings)

        pos_score = tf.exp(ttl_score) * other_mask
        ttl_score = ttl_score - tf.log(tf.reduce_sum(pos_score, axis=1, keepdims=True) + 1e-10)

        ssl_loss = -tf.reduce_mean(tf.reduce_sum(mask * ttl_score, axis=1) / (tf.reduce_sum(mask, axis=1) + 1e-10))

        return reg * ssl_loss

    def target_attention(self,
                         query,
                         values,
                         lens,
                         num_units,
                         name):

        max_len = values.shape[1]
        hidden_dim = values.shape[-1]

        query = tf.tile(query, [1, max_len])
        query = tf.reshape(query, [-1, max_len, hidden_dim])

        concat = tf.concat([query, values, query - values, query * values], 2)

        attention_hidden_units = [num_units, 1]

        for i in range(len(attention_hidden_units)):
            activation = None if i == len(attention_hidden_units) - 1 else tf.nn.tanh
            outputs = tf.layers.dense(concat, attention_hidden_units[i], activation=activation,
                                      name='attention_%s_%i' % (name, i))
            concat = outputs

        outputs = tf.reshape(outputs, [-1, 1, max_len])

        mask = tf.sequence_mask(lens, max_len)  # (batch_size, 1, max_len)
        # mask = ~mask
        padding = tf.ones_like(outputs) * (-1e12)
        outputs = tf.where(mask, outputs, padding)

        # 对outputs进行scale
        outputs = outputs / (int(hidden_dim) ** 0.5)
        outputs = tf.nn.softmax(outputs)

        outputs = tf.matmul(outputs, values)  # batch_size, 1, hidden_units)
        outputs = tf.squeeze(outputs, [1])  # (batch_size, hidden_units)
        return outputs

    @property
    def name(self):
        return 'BARL-ASe'