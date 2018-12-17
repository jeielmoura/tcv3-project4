import tensorflow as tf
import numpy as np
import random
import time
import sys
import os

from constants import *
from data import *
from train import *

classes = sorted(os.listdir(TRAIN_FOLDER))
images = [sorted(os.listdir(TRAIN_FOLDER + '/' + id)) for id in classes]
NUM_IMAGES = np.sum([len(l) for l in images])

NNS_VERSION = np.array(['0', '1', '2', '3', '4'])
NUM_NNS = len(NNS_VERSION)
NNS_PATH = np.array( [ (MODEL_FOLDER + '/' + 'model' + v + '.ckpt') for v in NNS_VERSION ] )


NNS_USED = np.zeros( (NUM_NNS,), dtype=np.uint8)

NNS_USED_BEST = np.zeros( (NUM_NNS,), dtype=np.uint8)
best_val_acc = -1.
best_val_loss = -1.

def combine_evaluation (sess) :

	if np.sum(NNS_USED) == 0 :
		return 0., -1.

	train_ans = np.ones( (len(X_ensem),len(classes_train)), dtype=np.uint8)/1.
	
	saver = tf.train.Saver( var_list=tf.trainable_variables() )
	for i in range (NUM_NNS) :
		if NNS_USED[i] :
			saver.restore(sess, NNS_PATH[i])
			for j in range( len(X_ensem) ) :
				ret = sess.run([probs], feed_dict = { X: np.expand_dims(X_ensem[j], axis=0), is_training: False })
				train_ans[j] *= ret[0].reshape(-1)		

	eval_loss = 0.
	eval_acc = 0.
	for j in range( len(X_ensem) ) :
		resp = np.argmax(train_ans[j])
		eval_acc += (resp == y_ensem[j])
		eval_loss += (resp - y_ensem[j])**2

	return eval_acc/len(X_ensem), eval_loss/len(X_ensem)

def recursion (sess, i=0) :
	global best_val_acc, best_val_loss
	global NNS_USED, NNS_USED_BEST

	if (i < NUM_NNS) :
		if np.random.randint(2) :
			NNS_USED[i] = 0
			recursion(sess, i+1)
			NNS_USED[i] = 1
			recursion(sess, i+1)
		else :
			NNS_USED[i] = 1
			recursion(sess, i+1)
			NNS_USED[i] = 0
			recursion(sess, i+1)
	
	if (i == NUM_NNS) and (np.sum(NNS_USED) > 0) :
		val_acc, val_loss = combine_evaluation(sess)
		print(np.sum(NNS_USED), ':', NNS_USED, "ACC:", val_acc, "Loss:", val_loss)

		better = (best_val_acc <= val_acc)
		if best_val_acc == val_acc :
			better = (best_val_loss > val_loss)

		if better :
			NNS_USED_BEST = np.array(NNS_USED)
			best_val_loss = val_loss
			best_val_acc = val_acc
			best_nn_num = np.sum(NNS_USED)
			print("better")


def combine_save (sess) :
	if np.sum(NNS_USED_BEST) < 1 :
		return

	NUM_IMAGES = len( os.listdir(TEST_FOLDER) )
	
	VERSION_NAME = '[ '
	for i in range (NUM_NNS) :
		if NNS_USED_BEST[i] :
			VERSION_NAME = VERSION_NAME + NNS_VERSION[i] + ' '
	VERSION_NAME = VERSION_NAME + ']'

	RESULT_PATH = RESULT_FOLDER + '/result='+ VERSION_NAME +'.txt'

	test_ans = np.ones( (NUM_IMAGES,len(classes_train)), dtype=np.uint8)/1.
	T, name = load_test_dataset()
	T = T/255.

	saver = tf.train.Saver( var_list=tf.trainable_variables() )
	for i in range (NUM_NNS) :
		if NNS_USED_BEST[i] :
			saver.restore(sess, NNS_PATH[i])
			for t in range( len(T) ) :
				ret = sess.run([probs], feed_dict = { X: np.expand_dims(T[t], axis=0), is_training: False })
				test_ans[t] *= ret[0].reshape(-1)

	file = open(RESULT_PATH, "w")
	file.write(str(best_val_acc) + ' ' + str(best_val_loss) + '\n\n')
	for t in range( len(T) ) :	
		file.write(name[t] + ' ' + classes_train[ np.argmax(test_ans[t]) ] + '\n')
	file.close()

def combine () :
	global NNS_USED_BEST

	with tf.Session(graph = graph) as session:
		session.run(tf.global_variables_initializer())
		recursion(session)

		print('melhor =', NNS_USED_BEST, best_val_acc, best_val_loss)
		combine_save(session)


def main () :
	#train_model('0')
	#train_model('1')
	#train_model('2')
	#train_model('3')
	train_model('4')

	combine()

if __name__ == '__main__' :
	main()