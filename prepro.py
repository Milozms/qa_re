import json
import pickle
import linecache
from tqdm import tqdm
import re
import numpy as np
from nltk.tokenize import WordPunctTokenizer

def build_word_list():
	word_tokenizer = WordPunctTokenizer()
	wordset = set()
	files = ['./data/sq_relations/test.replace_ne.withpool',
			 './data/sq_relations/train.replace_ne.withpool',
			 './data/sq_relations/valid.replace_ne.withpool']
	for infile in files:
		for line in linecache.getlines(infile):
			line = line.strip('\n')
			tokens = line.split('\t')
			question = tokens[-1]
			words = word_tokenizer.tokenize(question)
			for word in words:
				wordset.add(word)
	maxrellen = 0
	for line in linecache.getlines('./data/sq_relations/relation.2M.list'):
		relname = re.split('[/_]', line.strip())[1:]
		for word in relname:
			wordset.add(word)
		if len(relname) > maxrellen:
			maxrellen = len(relname)
	print('Size of wordset: %d' % len(wordset))
	print('Max relation length: %d' % maxrellen)
	with open('./data/dicts/wordlist.json', 'w') as f:
		json.dump(list(wordset), f)
	# Size of wordset: 9220
	# Max relation length: 17

def build_word_dict_emb():
	dim = 300
	with open('./data/dicts/wordlist.json', 'r') as f:
		wordlist = json.load(f)
	word2id = {}
	for i, word in enumerate(wordlist):
		word2id[word] = i
	vocab_size = len(wordlist)
	emb = np.zeros([vocab_size, dim])
	initialized = {}
	pretrained = 0
	avg_sigma = 0
	avg_mu = 0
	for line in tqdm(linecache.getlines('/Users/zms/Documents/学习资料/NLP/glove.840B.300d.txt')):
		line = line.strip()
		tokens = line.split()
		word = tokens[0]
		if word in word2id:
			vec = np.array([float(tok) for tok in tokens[-dim:]])
			wordid = word2id[word]
			emb[wordid] = vec
			initialized[word] = True
			pretrained += 1
			mu = vec.mean()
			sigma = np.std(vec)
			avg_mu += mu
			avg_sigma += sigma
	avg_sigma /= 1. * pretrained
	avg_mu /= 1. * pretrained
	for w in word2id:
		if w not in initialized:
			emb[word2id[w]] = np.random.normal(avg_mu, avg_sigma, (dim,))
	print(pretrained, vocab_size)
	# 8361 9220
	with open('./data/dicts/wordemb.pickle', 'wb') as f:
		pickle.dump(emb, f)
	with open('./data/dicts/word2id.json', 'w') as f:
		json.dump(word2id, f)


if __name__ == '__main__':
	build_word_list()
	build_word_dict_emb()
