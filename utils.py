import numpy as np
import pickle
import linecache
import re
import json
from tqdm import tqdm
from nltk.tokenize import WordPunctTokenizer

def pad_sequence(seq, maxlen):
	if len(seq)>maxlen:
		return seq[:maxlen]
	pad_len = maxlen - len(seq)
	return seq + [0]*pad_len

def read_kb_emb():
	kb2id = {}
	dim = 100
	initialized = {}
	pretrained = 0
	emb_whole = []
	print('Reading relation embedding......')
	with open('./data/fb2m/relation2vecfb2m.vec', 'r') as f:
		for line in tqdm(f.readlines()):
			line = line.strip()
			tokens = line.split()
			emb_whole.append([float(x) for x in tokens])
	emb = np.array(emb_whole)
	return emb

class Dataset(object):
	def __init__(self, filename, max_cnt = None, shuffle = True):
		word_tokenizer = WordPunctTokenizer()
		with open('./data/dicts/word2id.json', 'r') as f:
			word2id = json.load(f)
		with open('./data/sq_relations/relation.2M.list', 'r') as f:
			rel_list = f.readlines()
		relations = []
		relnames = []
		negative_relations = []
		negative_relnames = []
		questions = []
		maxlen = 0
		maxrellen = 0
		max_neg_cnt = 0
		for line in linecache.getlines(filename):
			line = line.strip()
			tokens = line.split('\t')
			try:
				relation = int(tokens[0])
			except:
				print('Relation more than one: %s' % tokens[0])
			nrelation = tokens[1].split(' ')
			nrelation = [int(r) for r in nrelation]
			question = tokens[-1]
			words_ = word_tokenizer.tokenize(question)
			words = []
			for word in words_:
				if word != '' and word != '#':
					try:
						word_id = word2id[word]
						words.append(word_id)
					except:
						print('%s not exist' % word)
			relname = re.split('[/_]', rel_list[relation].strip())[1:]
			relwords = []
			for word in relname:
				if word != '':
					try:
						word_id = word2id[word]
						relwords.append(word_id)
					except:
						print('%s not exist' % word)
			questions.append(words)
			relations.append(relation)
			relnames.append(relwords)
			negative_relations.append(nrelation)
			if len(words) > maxlen:
				maxlen = len(words)
			if len(relwords) > maxrellen:
				maxrellen = len(relwords)
			if len(nrelation) > max_neg_cnt:
				max_neg_cnt = len(nrelation)
		self.data = []
		self.datasize = len(questions)
		if max_cnt != None and max_cnt < self.datasize:
			self.datasize = max_cnt
		for i in range(self.datasize):
			self.data.append((questions[i], relations[i], relnames[i], len(questions[i]), len(relnames[i]),
							  negative_relations[i]))
		self.maxlen = maxlen
		self.maxrellen = maxrellen
		self.max_neg_cnt = max_neg_cnt
		if shuffle:
			np.random.shuffle(self.data)
		self.current_index = 0

	def get_mini_batch(self, batch_size):
		if self.current_index >= self.datasize:
			return None
		if self.current_index + batch_size > self.datasize:
			batch = self.data[self.current_index:]
			self.current_index = self.datasize
		else:
			batch = self.data[self.current_index:self.current_index + batch_size]
			self.current_index += batch_size
		relations = []
		relnames = []
		questions = []
		qlen = []
		rellen = []
		for ins in batch:
			questions.append(ins[0])
			relations.append(ins[1])
			relnames.append(ins[2])
			qlen.append(ins[3])
			rellen.append(ins[4])
		return questions, relations, relnames, qlen, rellen





