import tensorflow as tf
from utils import Dataset, read_kb_emb
from tqdm import tqdm
import nltk
import logging
import pickle
import numpy as np
import json
import os
import math

class Model(object):
	def __init__(self, config, kb_emb_mat, word_emb_mat):
		self.hidden = config.hidden
		self.word_vocab_size = config.word_vocab_size
		self.word_emb_dim = config.word_emb_dim
		self.kb_emb_dim = config.kb_emb_dim
		self.batch = config.batch
		self.is_train = config.is_train
		self.maxlen = config.maxlen
		self.maxrellen = config.maxrellen
		self.max_neg_cnt = config.max_neg_cnt
		self.word_emb_mat = word_emb_mat
		self.kb_emb_mat = kb_emb_mat
		self.epoch_num = config.epoch_num
		self.max_grad_norm = config.max_grad_norm
		self.lr = config.lr
		self.maxbleu = 0.0
		self.minloss = 100
		self.build()

	def residual_lstm(self, hidden, inputs, seq_len):
		with tf.variable_scope('resBiLSTM'):
			outputs = []
			for layer in range(2):
				with tf.variable_scope("Layer_{}".format(layer)):
					cell_fw = tf.nn.rnn_cell.BasicLSTMCell(hidden)
					cell_bw = tf.nn.rnn_cell.BasicLSTMCell(hidden)
					init_fw = cell_fw.zero_state(self.batch, dtype=tf.float32)
					init_bw = cell_bw.zero_state(self.batch, dtype=tf.float32)
					output, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell_fw, cell_bw=cell_bw,
																inputs=inputs, sequence_length=seq_len,
																initial_state_fw=init_fw, initial_state_bw=init_bw,
																dtype=tf.float32)
					output = tf.concat([output[0], output[1]], axis=2)
					outputs.append(output)
			outputs = tf.stack(outputs, axis=0)
			outputs = tf.reduce_sum(outputs, axis=0)
			# max pooling
			outputs = tf.reduce_max(outputs, axis=1)
			return outputs

	def relation_lstm(self, hidden, inputs, seq_len, rel_emb, question):
		with tf.variable_scope('BiLSTM_relation', reuse=tf.AUTO_REUSE):
			# relation word representation
			rel_cell_fw = tf.nn.rnn_cell.BasicLSTMCell(hidden)
			rel_cell_bw = tf.nn.rnn_cell.BasicLSTMCell(hidden)
			init_fw = rel_cell_fw.zero_state(self.batch, dtype=tf.float32)
			init_bw = rel_cell_bw.zero_state(self.batch, dtype=tf.float32)
			output, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=rel_cell_fw, cell_bw=rel_cell_bw,
											inputs=inputs, sequence_length=seq_len,
											initial_state_fw=init_fw, initial_state_bw=init_bw, dtype=tf.float32)
			output = tf.concat([output[0], output[1]], axis=2)
			# max pooling
			rel_words = tf.reduce_max(output, axis=1)

			# concat relation embedding and relation words
			relation = tf.concat([rel_emb, rel_words], axis=1)

			rel_dim = hidden*2 + self.kb_emb_dim
			ques_dim = hidden*2
			weight = tf.get_variable('weight', shape=[rel_dim, ques_dim], dtype=tf.float32,
									 initializer=tf.random_normal_initializer())

			# q^T*W*r
			relation = tf.matmul(relation, weight)
			score = tf.reduce_sum(tf.multiply(question, relation), axis=1)

			return score


	def build(self):
		self.question = tf.placeholder(dtype=tf.int32, shape=[None, self.maxlen], name='question')
		self.qlen = tf.placeholder(dtype=tf.int32, shape=[None], name='question_len')
		self.relid = tf.placeholder(dtype=tf.int32, shape=[None], name='relation_id')
		self.relname = tf.placeholder(dtype=tf.int32, shape=[None, self.maxrellen], name='relation_name')
		self.rellen = tf.placeholder(dtype=tf.int32, shape=[None], name='relation_len')
		# negative samples
		self.neg_rels = tf.placeholder(dtype=tf.int32, shape=[None, self.max_neg_cnt], name='neg_relation_ids')
		self.neg_rel_lens = tf.placeholder(dtype=tf.int32, shape=[None, self.max_neg_cnt], name='neg_relation_lens')
		self.neg_rel_names = tf.placeholder(dtype=tf.int32, shape=[None, self.max_neg_cnt, self.maxrellen],
											name='neg_relation_names')
		self.neg_cnt = tf.placeholder(dtype=tf.int32, shape=[None], name='neg_cnt')

		self.keep_prob = tf.placeholder(dtype=tf.float32, shape=())
		batch_size = tf.shape(self.question)[0]
		# batch_size = self.triple.shape[0].value

		hidden = self.hidden
		word_vocab_size = self.word_vocab_size
		word_emb_dim = self.word_emb_dim
		kb_emb_dim = self.kb_emb_dim
		maxlen = self.maxlen
		maxrellen = self.maxrellen

		with tf.device("/cpu:0"):
			with tf.variable_scope("embeddings"):
				word_embeddings = tf.get_variable(name = "word_embedding",
									dtype = tf.float32,
									initializer = tf.constant(self.word_emb_mat, dtype=tf.float32),
									trainable=False)
				kb_embeddings = tf.get_variable(name = "kb_embedding",
									dtype = tf.float32,
									initializer = tf.constant(self.kb_emb_mat, dtype=tf.float32),
									trainable=True)

		ques_emb = tf.nn.embedding_lookup(word_embeddings, self.question)
		rel_emb = tf.nn.embedding_lookup(kb_embeddings, self.relid)
		rel_word_emb = tf.nn.embedding_lookup(word_embeddings, self.relname)

		question = self.residual_lstm(hidden, ques_emb, self.qlen)
		score = self.relation_lstm(hidden, rel_word_emb, self.rellen, rel_emb, question)

		# TODO:negative samples



		return


	def train(self, dset, valid_dset):
		saver = tf.train.Saver()
		tfconfig = tf.ConfigProto()
		# tfconfig.gpu_options.allow_growth = True
		sess = tf.Session(config=tfconfig)
		sess.run(tf.global_variables_initializer())
		num_batch = int(dset.datasize / self.batch) + 1
		for ei in range(self.epoch_num):
			dset.current_index = 0
			loss_iter = 0.0
			for bi in tqdm(range(num_batch)):
				mini_batch = train_dset.get_mini_batch(self.batch)
				if mini_batch == None:
					break
				triples, questions, qlen, subnames = mini_batch
				feed_dict = {}
				feed_dict[self.triple] = triples
				feed_dict[self.question] = questions
				feed_dict[self.qlen] = qlen
				feed_dict[self.keep_prob] = 1.0
				loss, train_op, out_idx = sess.run(self.out, feed_dict=feed_dict)
				loss_iter += loss
			loss_iter /= num_batch
			logging.info('iter %d, train loss: %f' % (ei, loss_iter))
			self.valid_model(sess, valid_dset, ei, saver)
			# mtest.test_model(sess, test_dset, ei, saver)


if __name__ == '__main__':
	os.environ["CUDA_VISIBLE_DEVICES"] = '0'
	logger = logging.getLogger()
	logger.setLevel(logging.INFO)
	handler = logging.FileHandler("./log/log.txt", mode='w')
	handler.setLevel(logging.INFO)
	console = logging.StreamHandler()
	console.setLevel(logging.INFO)
	formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
	handler.setFormatter(formatter)
	console.setFormatter(formatter)
	logger.addHandler(handler)
	logger.addHandler(console)
	with open('./data/dicts/word2id.json', 'rb') as f:
		word2id = json.load(f)
	with open('./data/dicts/wordemb.pickle', 'rb') as f:
		wordemb = pickle.load(f)
	kbemb = read_kb_emb()
	flags = tf.flags
	flags.DEFINE_integer('hidden', 400, "")
	flags.DEFINE_integer('word_vocab_size', len(word2id), "")
	flags.DEFINE_integer('word_emb_dim', 300, "")
	flags.DEFINE_integer('kb_emb_dim', 100, "")
	flags.DEFINE_integer('maxlen', 35, "")
	flags.DEFINE_integer('maxrellen', 20, "")
	flags.DEFINE_integer('max_neg_cnt', 20, "")
	flags.DEFINE_integer('batch', 128, "")
	flags.DEFINE_integer('epoch_num', 200, "")
	flags.DEFINE_boolean('is_train', True, "")
	flags.DEFINE_float('max_grad_norm', 0.1, "")
	flags.DEFINE_float('lr', 0.00025, "")
	config = flags.FLAGS
	train_file = './sq/annotated_fb_data_train.txt'
	valid_file = './sq/annotated_fb_data_valid.txt'
	train_dset = Dataset(train_file)
	valid_dset = Dataset(valid_file)
	with tf.variable_scope('model'):
		model = Model(config, word_emb_mat=wordemb, kb_emb_mat=kbemb)
	# config.is_train = False
	# with tf.variable_scope('model', reuse=True):
	# 	mtest = Model(config, word_emb_mat=wordemb, kb_emb_mat=kbemb)
	# model.train(train_dset, valid_dset)



