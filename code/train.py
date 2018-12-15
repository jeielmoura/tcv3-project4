import tensorflow as tf
import numpy as np
import random
import time
import sys
import os

from constants import *
from data import *

# ---------------------------------------------------------------------------------------------------------- #
# Description:                                                                                               #
#         Parameters.                                                                                        #
# ---------------------------------------------------------------------------------------------------------- #


# ---------------------------------------------------------------------------------------------------------- #
# Description:                                                                                               #
#         Load the training set, shuffle its images and then split them in training and validation subsets.  #
#         After that, load the testing set.                                                                  #
# ---------------------------------------------------------------------------------------------------------- #


# ---------------------------------------------------------------------------------------------------------- #
# Description:                                                                                               #
#         Create a training graph that receives a batch of images and their respective labels and run a      #
#         training iteration or an inference job. Train the last FC layer using fine_tuning_op or the entire #
#         network using full_backprop_op. A weight decay of 1e-4 is used for full_backprop_op only.          #
# ---------------------------------------------------------------------------------------------------------- #
graph = tf.Graph()
with graph.as_default():
	X = tf.placeholder(tf.float32, shape = (None, IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS))
	y = tf.placeholder(tf.int64, shape = (None,))
	y_one_hot = tf.one_hot(y, len(classes_train))
	learning_rate = tf.placeholder(tf.float32)
	is_training = tf.placeholder(tf.bool)
	
	print(X.shape)

	out = tf.layers.max_pooling2d(X, (2, 2), (2, 2), padding='same')
	
	out = tf.layers.conv2d(out, 32, (2, 2), (1, 1), padding='same', activation=tf.nn.relu)

	out = tf.layers.dropout(out, rate=0.4, training=is_training)

	out = tf.layers.conv2d(out, 16, (2, 2), (1, 1), padding='same', activation=tf.nn.relu)

	out = tf.layers.dropout(out, rate=0.4, training=is_training)
	

	out = tf.reshape(out, [-1, out.shape[1]*out.shape[2]*out.shape[3]])

	out = tf.layers.dropout(out, rate=0.4, training=is_training)

	out = tf.layers.dense(out, 128, activation=tf.nn.relu)

	out = tf.layers.dropout(out, rate=0.4, training=is_training)

	#out = tf.layers.dense(out, len(classes_train), activation=tf.nn.sigmoid)
	out = tf.layers.dense(out, len(classes_train))

	
	#loss = tf.reduce_mean(tf.reduce_sum((y_one_hot-out)**2))
	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=out, labels=y_one_hot))


	train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

	probs = tf.nn.softmax(out)
	result = tf.argmax(probs, 1)
	correct = tf.reduce_sum(tf.cast(tf.equal(result, y), tf.float32))

# ---------------------------------------------------------------------------------------------------------- #
# Description:                                                                                               #
#         Run one training epoch using images in X_train and labels in y_train.                              #
# ---------------------------------------------------------------------------------------------------------- #
def training_epoch(session, op, lr, epoch):
	batch_list = np.random.permutation(len(X_train))

	start = time.time()
	train_loss = 0
	train_acc = 0

	for j in range(0, len(X_train), BATCH_SIZE):
		if j+BATCH_SIZE > len(X_train):
			break
		X_batch = X_train.take(batch_list[j:j+BATCH_SIZE], axis=0)
		y_batch = y_train.take(batch_list[j:j+BATCH_SIZE], axis=0)

		ret = session.run([op, loss, correct], feed_dict = {X: argumentation(X_batch), y: y_batch, learning_rate: lr, is_training: True})
		train_loss += ret[1]*BATCH_SIZE
		train_acc += ret[2]

	pass_size = (len(X_train)-len(X_train)%BATCH_SIZE)
	print('Epoch:'+str(epoch)+' Train: Time:'+str(time.time()-start)+' ACC:'+str(100*train_acc/pass_size)+' Loss:'+str(train_loss/pass_size)+' LR:'+str(lr))
# ---------------------------------------------------------------------------------------------------------- #
# Description:                                                                                               #
#         Evaluate images in Xv with labels in yv.                                                           #
# ---------------------------------------------------------------------------------------------------------- #
def evaluation(session, Xv, yv, epoch):
	start = time.time()
	eval_loss = 0
	eval_acc = 0
	for j in range(0, len(Xv), BATCH_SIZE):
		ret = session.run([loss, correct], feed_dict = {X: Xv[j:j+BATCH_SIZE], y: yv[j:j+BATCH_SIZE], is_training: False})
		eval_loss += ret[0]*min(BATCH_SIZE, len(Xv)-j)
		eval_acc += ret[1]

	print('Epoch:'+str(epoch)+' Valid: Time:'+str(round(time.time()-start,5))+' ACC:'+str(round(100*eval_acc/len(Xv),9))+' Loss:'+str(round(eval_loss/len(Xv),9)))

	return eval_acc/len(Xv), eval_loss/len(Xv)



# ---------------------------------------------------------------------------------------------------------- #
# Description:                                                                                               #
#         Training loop and execution.																		 #
# ---------------------------------------------------------------------------------------------------------- #	
def train_model (model_version) :
	global X_train, y_train
	global X_val, y_val
	global X_ensem, y_ensem

	global LEARNING_RATES, LEARNING_RATE_DECAY

	X_train, y_train = shuffle(X_train, y_train)
	X_train, y_train, X_val, y_val = split(X_train, y_train, TRAIN_SPLIT_RATE)

	X_val = np.concatenate( (X_val, X_ensem), axis=0)
	y_val = np.concatenate( (y_val, y_ensem), axis=0)

	LEARNING_RATES_ARRAY = np.array(LEARNING_RATES)

	MODEL_PATH = MODEL_FOLDER + '/' + 'model' + model_version + '.ckpt'
	RESULT_PATH = MODEL_FOLDER + '/' + 'result' + model_version + '.txt'
	
	with tf.Session(graph = graph) as session:
		session.run(tf.global_variables_initializer())

		best_val_acc = -1.
		best_val_loss = -1.

		saver = tf.train.Saver( var_list=tf.trainable_variables() )

		epoch = 1
		while epoch <= NUM_EPOCHS_LIMIT :
			updated = False
			step = 0.1
			for LR in LEARNING_RATES_ARRAY :
				X_train, y_train = shuffle(X_train, y_train)

				training_epoch(session, train_op, LR, epoch + step)
				val_acc, val_loss = evaluation(session, X_val, y_val, epoch + step)

				better = (best_val_acc <= val_acc and best_val_loss >= val_loss)
				better = better and (best_val_acc < val_acc or best_val_loss > val_loss)
				better = better or (best_val_acc < 0 and best_val_loss < 0)

				if better :
					updated = True
					best_val_acc = val_acc
					best_val_loss = val_loss
					saver.save(session, MODEL_PATH)
					print('ACC:'+str(100*best_val_acc)+' Loss:'+str(best_val_loss))
				step += 0.1

			LEARNING_RATES_ARRAY *= LEARNING_RATE_DECAY
			
			if updated :
				epoch = 1
				print()
				print()
			else :
				print('------------')
				epoch += 1

			

		saver.restore(session, MODEL_PATH)

		print('ACC:'+str(100*best_val_acc)+' Loss:'+str(best_val_loss))
		
		
		T, name = load_test_dataset()
		T = T/255.

		file = open(RESULT_PATH, "w")
		file.write(str(100*best_val_acc) + ' ' + str(best_val_loss) + '\n\n')
		for t in range( len(T) ) :  
			ret = session.run([result], feed_dict = { X: np.expand_dims(T[t], axis=0), is_training: False})
			file.write(name[t] + ' ' + classes_train[ int(ret[0]) ] + '\n')
		file.close()
