'''
-------------------------------------
FUNCTION DEFINITIONS FOR ARCHITECTURE
-------------------------------------
'''

import tensorflow as tf
from tensorflow.contrib import rnn

import collections
import time
import random
import codecs
import os
import numpy as np

import parameters
import additional_functions as helper_fns

__all__=['embedding','create_or_load_model','load_model','run_full_eval']

def embedding(src_vocab_size,tgt_vocab_size,inembsize) : 

	#print 'Embedding Function'

	embedding_encoder=tf.get_variable(name='embedding_encoder',shape=[src_vocab_size,inembsize],dtype=tf.float32)
	embedding_decoder=tf.get_variable(name='embedding_decoder',shape=[tgt_vocab_size,inembsize],dtype=tf.float32)

	return [embedding_encoder,embedding_decoder]

def load_model(model,ckpt,sess,name) : 
	start_time=time.time()
	model.saver.restore(sess,ckpt)
	sess.run(tf.tables_initializer())
	return model

def create_or_load_model(model,model_dir,sess,name) : 
	latest_ckpt=tf.train.latest_checkpoint(model_dir)
	if latest_ckpt : 
		print 'model obtained from checkpoint'
		model=load_model(model,latest_ckpt,sess,name)
	else : 
		print 'model created'
		start=time.time()
		sess.run(tf.global_variables_initializer())
		sess.run(tf.tables_initializer())

	global_step=model.global_step.eval(session=sess)
	return [model,global_step]

def sample_decode(model,global_step,sess,param,iterator,src_data,tgt_data,iterator_src_placeholder,
	iterator_batch_size_placeholder,summary_writer) : 
	# Pick a random sentence
	decode_id=random.randint(0,len(src_data)-1)

	iterator_feed_dict={iterator_src_placeholder:[src_data[decode_id]],iterator_batch_size_placeholder:1}

	sess.run(iterator.initializer,feed_dict=iterator_feed_dict)

	nmt_output,attention_summary=model.decode(sess)

	translation=helper_fns.translate(nmt_output,
		sent_id=0,tgt_eos='</s>')

	# Printing translation
	print 'Source : ',src_data[decode_id]
	print 'Actual Target : ',tgt_data[decode_id]
	print 'Translation : ',translation

def run_full_eval(model_dir,infer_model,infer_sess,
	eval_model,eval_sess,param,summary_writer,src_data,tgt_data) : 
	
	# Decode a random sentence from source data
	with infer_model.graph.as_default() : 
		loaded_infer_model,global_step=create_or_load_model(infer_model.model,model_dir,infer_sess,'infer')

	sample_decode(loaded_infer_model,global_step,infer_sess,param,infer_model.iterator,src_data,tgt_data,
		infer_model.src_placeholder,infer_model.batch_size_placeholder,summary_writer)

def decode_and_evaluate(model,global_step,sess,param,iterator,
		iterator_feed_dict,ref_file,label) : 
	
	out_dir=param.out_dir

	if global_step==0 : 
		return 0

	print 'Evaluation at global_step=',global_step

	output=trans_file=os.path.join(out_dir,'output_%s.txt'%label)

	start_time=time.time()
	num_sentences=0

	sess.run(iterator.initializer,feed_dict=iterator_feed_dict)

	with codecs.getwriter('utf-8')(tf.gfile.GFile(trans_file,mode='wb')) as trans_f : 
		trans_f.write(' ')

		while True : 
			try : 
				nmt_output,_=model.decode(sess)
				nmt_output=np.expand_dims(nmt_output,0)
				batch_size=nmt_output.shape[1]
				num_sentences+=batch_size

				for sent_id in range(batch_size) : 
					translation=helper_fns.translate(nmt_output[0],sent_id,tgt_eos='</s>')
					#print 'Test Translation : ',translation
					#print type(translation)
					trans_f.write((translation+'\n').decode('utf-8'))
					#trans_f.write('\n')

			except tf.errors.OutOfRangeError : 
				print num_sentences,' sentences decoded.'
				break


	os.system('python calculate_bleu_score.py '+ref_file+' '+trans_file)















