'''
--------------------------------------
LIST OF PARAMETERS AND HYPERPARAMETERS
--------------------------------------
'''
import tensorflow as tf

def create_params_hparams() : 

	return tf.contrib.training.HParams(
		
		batch_size=128,
		infer_batch_size=32,
		learning_rate=0.01,
		init_weight=0.1,
		max_grad_norm=5.0,
		init_method='uniform',
		num_train_steps=1000,

		num_lstm_units=512,
		num_layers_encoder=2,
		num_layers_decoder=2,
		inembsize=256, # size of embedding
		dropout=0.7, # during training
		
		steps_per_stats=100,
		epoch_step=0,
		
		attention_option='luong',
		output_attention=True,
		pass_hidden_state=True, # to pass encoder's final state to decoder's initial state
		time_major=True,
		
		out_dir='En_Vi_Out_Dir',
		
		src_max_len=50, # max length of source sequence during training 
		tgt_max_len=50, # max length of target sequence during training
		src_max_len_inf=0, # max length of source sequence during testing
		tgt_max_len_inf=0, # max length of target sequence during testing
		
		inf_input_file='tst2013.en',
		inf_output_file='tst2013.vi',

		sampling_temp=0.0, # used for inference, 0.0 means greedy decoding

		num_ckpts=5, # number of checkpoints to keeps
		ckpt=None
		)
