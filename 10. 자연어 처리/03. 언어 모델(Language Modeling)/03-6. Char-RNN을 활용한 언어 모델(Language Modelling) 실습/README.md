# TensorFlow 2.0을 이용한 Char-RNN 구현

## 필요한 라이브러리 import

```python
from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

import numpy as np
import os
import time
```

## 유틸리티 함수 정의

```python
# input 데이터와 input 데이터를 한글자씩 뒤로 민 target 데이터를 생성하는 utility 함수를 정의합니다.
def split_input_target(chunk):
  input_text = chunk[:-1]
  target_text = chunk[1:]

  return input_text, target_text
```

## 설정값 지정

```python
# 학습에 필요한 설정값들을 지정합니다.
#data_dir = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')  # shakespeare
data_dir = tf.keras.utils.get_file('linux.txt', 'https://raw.githubusercontent.com/solaris33/deep-learning-tensorflow-book-code/master/Ch08-RNN/Char-RNN/data/linux/input.txt')  # linux
batch_size = 64      # Training : 64, Sampling : 1
seq_length = 100     # Training : 100, Sampling : 1
embedding_dim = 256  # Embedding 차원
hidden_size = 1024   # 히든 레이어의 노드 개수
num_epochs = 10
```

```
Downloading data from https://raw.githubusercontent.com/solaris33/deep-learning-tensorflow-book-code/master/Ch08-RNN/Char-RNN/data/linux/input.txt
6209536/6206996 [==============================] - 0s 0us/step
```

## 어휘 집합(Vocabulary set) 설정

```python
# 학습에 사용할 txt 파일을 읽습니다.
text = open(data_dir, 'rb').read().decode(encoding='utf-8')
# 학습데이터에 포함된 모든 character들을 나타내는 변수인 vocab과
# vocab에 id를 부여해 dict 형태로 만든 char2idx를 선언합니다.
vocab = sorted(set(text))  # 유니크한 character 개수
vocab_size = len(vocab)
print('{} unique characters'.format(vocab_size))
char2idx = {u: i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)
```

```
99 unique characters
```

## Dataset 설정

```python
# 학습 데이터를 character에서 integer로 변환합니다.
text_as_int = np.array([char2idx[c] for c in text])

# split_input_target 함수를 이용해서 input 데이터와 input 데이터를 한글자씩 뒤로 민 target 데이터를 생성합니다.
char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)
sequences = char_dataset.batch(seq_length+1, drop_remainder=True)
dataset = sequences.map(split_input_target)

# tf.data API를 이용해서 데이터를 섞고 batch 형태로 가져옵니다.
dataset = dataset.shuffle(10000).batch(batch_size, drop_remainder=True)
```

## RNN 모델 설정

```python
# tf.keras.Model을 이용해서 RNN 모델을 정의합니다.
class RNN(tf.keras.Model):
 def __init__(self, batch_size):
   super(RNN, self).__init__()
   self.embedding_layer = tf.keras.layers.Embedding(vocab_size, embedding_dim,
                                                    batch_input_shape=[batch_size, None])
   self.hidden_layer_1 = tf.keras.layers.LSTM(hidden_size,
                                             return_sequences=True,
                                             stateful=True,
                                             recurrent_initializer='glorot_uniform')
   self.output_layer = tf.keras.layers.Dense(vocab_size)

 def call(self, x):
   embedded_input = self.embedding_layer(x)
   features = self.hidden_layer_1(embedded_input)
   logits = self.output_layer(features)

   return logits
```

## Loss Function 정의

```python
# sparse cross-entropy 손실 함수를 정의합니다.
def sparse_cross_entropy_loss(labels, logits):
  return tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True))
```

## 옵티마이저 및 학습 설정

```python
# 최적화를 위한 Adam 옵티마이저를 정의합니다.
optimizer = tf.keras.optimizers.Adam()

# 최적화를 위한 function을 정의합니다.
@tf.function
def train_step(model, input, target):
  with tf.GradientTape() as tape:
    logits = model(input)
    loss = sparse_cross_entropy_loss(target, logits)
  grads = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(grads, model.trainable_variables))

  return loss
```

## 샘플링 함수 설정

```python
def generate_text(model, start_string):
  num_sampling = 4000  # 생성할 글자(Character)의 개수를 지정합니다.

  # start_sting을 integer 형태로 변환합니다.
  input_eval = [char2idx[s] for s in start_string]
  input_eval = tf.expand_dims(input_eval, 0)

  # 샘플링 결과로 생성된 string을 저장할 배열을 초기화합니다.
  text_generated = []

  # 낮은 temperature 값은 더욱 정확한 텍스트를 생성합니다.
  # 높은 temperature 값은 더욱 다양한 텍스트를 생성합니다.
  temperature = 1.0

  # 여기서 batch size = 1 입니다.
  model.reset_states()
  for i in range(num_sampling):
    predictions = model(input_eval)
    # 불필요한 batch dimension을 삭제합니다.
    predictions = tf.squeeze(predictions, 0)

    # 모델의 예측결과에 기반해서 랜덤 샘플링을 하기위해 categorical distribution을 사용합니다.
    predictions = predictions / temperature
    predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()

    # 예측된 character를 다음 input으로 사용합니다.
    input_eval = tf.expand_dims([predicted_id], 0)
    # 샘플링 결과를 text_generated 배열에 추가합니다.
    text_generated.append(idx2char[predicted_id])

  return (start_string + ''.join(text_generated))
```

## 트레이닝 시작

```python
# Recurrent Neural Networks(RNN) 모델을 선언합니다.
RNN_model = RNN(batch_size=batch_size)

# 데이터 구조 파악을 위해서 예제로 임의의 하나의 배치 데이터 에측하고, 예측결과를 출력합니다.
for input_example_batch, target_example_batch in dataset.take(1):
  example_batch_predictions = RNN_model(input_example_batch)
  print(example_batch_predictions.shape, "# (batch_size, sequence_length, vocab_size)")

# 모델 정보를 출력합니다.
RNN_model.summary()

# checkpoint 데이터를 저장할 경로를 지정합니다.
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

for epoch in range(num_epochs):
  start = time.time()

  # 매 반복마다 hidden state를 초기화합니다. (최초의 hidden 값은 None입니다.)
  hidden = RNN_model.reset_states()

  for (batch_n, (input, target)) in enumerate(dataset):
    loss = train_step(RNN_model, input, target)

    if batch_n % 100 == 0:
      template = 'Epoch {} Batch {} Loss {}'
      print(template.format(epoch+1, batch_n, loss))

  # 5회 반복마다 파라미터를 checkpoint로 저장합니다.
  if (epoch + 1) % 5 == 0:
    RNN_model.save_weights(checkpoint_prefix.format(epoch=epoch))

  print ('Epoch {} Loss {:.4f}'.format(epoch+1, loss))
  print ('Time taken for 1 epoch {} sec\n'.format(time.time() - start))

RNN_model.save_weights(checkpoint_prefix.format(epoch=epoch))
print("트레이닝이 끝났습니다!")
```

```
(64, 100, 99) # (batch_size, sequence_length, vocab_size)
Model: "rnn_2"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
embedding_2 (Embedding)      multiple                  25344     
_________________________________________________________________
lstm_2 (LSTM)                multiple                  5246976   
_________________________________________________________________
dense_2 (Dense)              multiple                  101475    
=================================================================
Total params: 5,373,795
Trainable params: 5,373,795
Non-trainable params: 0
_________________________________________________________________
Epoch 1 Batch 0 Loss 4.595424652099609
Epoch 1 Batch 100 Loss 2.8029003143310547
Epoch 1 Batch 200 Loss 2.368008613586426
Epoch 1 Batch 300 Loss 1.9701776504516602
Epoch 1 Batch 400 Loss 1.740452527999878
Epoch 1 Batch 500 Loss 1.6691888570785522
Epoch 1 Batch 600 Loss 1.5763310194015503
Epoch 1 Batch 700 Loss 1.5131900310516357
Epoch 1 Batch 800 Loss 1.4898241758346558
Epoch 1 Batch 900 Loss 1.3949942588806152
Epoch 1 Loss 1.3979
Time taken for 1 epoch 64.18272733688354 sec

Epoch 2 Batch 0 Loss 1.8904311656951904
Epoch 2 Batch 100 Loss 1.4212713241577148
Epoch 2 Batch 200 Loss 1.387321949005127
Epoch 2 Batch 300 Loss 1.2630565166473389
Epoch 2 Batch 400 Loss 1.1293050050735474
Epoch 2 Batch 500 Loss 1.3102813959121704
Epoch 2 Batch 600 Loss 1.2361220121383667
Epoch 2 Batch 700 Loss 1.237483263015747
Epoch 2 Batch 800 Loss 1.2400585412979126
Epoch 2 Batch 900 Loss 1.20728600025177
Epoch 2 Loss 1.1243
Time taken for 1 epoch 64.56112098693848 sec

Epoch 3 Batch 0 Loss 1.2513527870178223
Epoch 3 Batch 100 Loss 1.125900387763977
Epoch 3 Batch 200 Loss 1.1476699113845825
Epoch 3 Batch 300 Loss 1.123602271080017
Epoch 3 Batch 400 Loss 1.0869883298873901
Epoch 3 Batch 500 Loss 1.0950208902359009
Epoch 3 Batch 600 Loss 1.1275570392608643
Epoch 3 Batch 700 Loss 1.195296049118042
Epoch 3 Batch 800 Loss 1.0688831806182861
Epoch 3 Batch 900 Loss 1.0584933757781982
Epoch 3 Loss 1.0435
Time taken for 1 epoch 81.9296350479126 sec

Epoch 4 Batch 0 Loss 1.1756036281585693
Epoch 4 Batch 100 Loss 1.0549757480621338
Epoch 4 Batch 200 Loss 1.0179253816604614
Epoch 4 Batch 300 Loss 1.0016586780548096
Epoch 4 Batch 400 Loss 1.034349799156189
Epoch 4 Batch 500 Loss 1.0481075048446655
Epoch 4 Batch 600 Loss 1.0366021394729614
Epoch 4 Batch 700 Loss 1.0327637195587158
Epoch 4 Batch 800 Loss 0.959072470664978
Epoch 4 Batch 900 Loss 1.0204027891159058
Epoch 4 Loss 1.0377
Time taken for 1 epoch 68.18256163597107 sec

Epoch 5 Batch 0 Loss 1.015885591506958
Epoch 5 Batch 100 Loss 1.0565637350082397
Epoch 5 Batch 200 Loss 0.992794930934906
Epoch 5 Batch 300 Loss 1.0025701522827148
Epoch 5 Batch 400 Loss 0.9452717304229736
Epoch 5 Batch 500 Loss 0.8931390643119812
Epoch 5 Batch 600 Loss 0.9416240453720093
Epoch 5 Batch 700 Loss 1.050402045249939
Epoch 5 Batch 800 Loss 0.9717079401016235
Epoch 5 Batch 900 Loss 0.9858691692352295
Epoch 5 Loss 0.9969
Time taken for 1 epoch 67.7448480129242 sec

Epoch 6 Batch 0 Loss 0.999776303768158
Epoch 6 Batch 100 Loss 0.9565187096595764
Epoch 6 Batch 200 Loss 0.8619758486747742
Epoch 6 Batch 300 Loss 0.8680692911148071
Epoch 6 Batch 400 Loss 0.848659873008728
Epoch 6 Batch 500 Loss 0.9475144743919373
Epoch 6 Batch 600 Loss 0.925066351890564
Epoch 6 Batch 700 Loss 0.9546129703521729
Epoch 6 Batch 800 Loss 0.9545434713363647
Epoch 6 Batch 900 Loss 0.881976306438446
Epoch 6 Loss 0.9614
Time taken for 1 epoch 67.80332779884338 sec

Epoch 7 Batch 0 Loss 1.013685703277588
Epoch 7 Batch 100 Loss 0.9058746099472046
Epoch 7 Batch 200 Loss 0.9408100247383118
Epoch 7 Batch 300 Loss 0.8668063282966614
Epoch 7 Batch 400 Loss 0.9071646332740784
Epoch 7 Batch 500 Loss 0.8915481567382812
Epoch 7 Batch 600 Loss 0.918425440788269
Epoch 7 Batch 700 Loss 0.9360130429267883
Epoch 7 Batch 800 Loss 0.8834318518638611
Epoch 7 Batch 900 Loss 0.9235985279083252
Epoch 7 Loss 0.8838
Time taken for 1 epoch 68.06172704696655 sec

Epoch 8 Batch 0 Loss 1.008132815361023
Epoch 8 Batch 100 Loss 0.9254401922225952
Epoch 8 Batch 200 Loss 0.921670138835907
Epoch 8 Batch 300 Loss 0.8934357166290283
Epoch 8 Batch 400 Loss 0.809149980545044
Epoch 8 Batch 500 Loss 0.9418428540229797
Epoch 8 Batch 600 Loss 0.8931186199188232
Epoch 8 Batch 700 Loss 0.8268086314201355
Epoch 8 Batch 800 Loss 0.8832077980041504
Epoch 8 Batch 900 Loss 0.8943663239479065
Epoch 8 Loss 0.8802
Time taken for 1 epoch 68.04621863365173 sec

Epoch 9 Batch 0 Loss 0.8887687921524048
Epoch 9 Batch 100 Loss 0.8788896203041077
Epoch 9 Batch 200 Loss 0.8118340969085693
Epoch 9 Batch 300 Loss 0.8165507316589355
Epoch 9 Batch 400 Loss 0.858704149723053
Epoch 9 Batch 500 Loss 0.8680853843688965
Epoch 9 Batch 600 Loss 0.8612232208251953
Epoch 9 Batch 700 Loss 0.8178400993347168
Epoch 9 Batch 800 Loss 0.9306390285491943
Epoch 9 Batch 900 Loss 0.8305807709693909
Epoch 9 Loss 0.8791
Time taken for 1 epoch 67.58101749420166 sec

Epoch 10 Batch 0 Loss 0.8652300238609314
Epoch 10 Batch 100 Loss 0.8335866332054138
Epoch 10 Batch 200 Loss 0.8167266845703125
Epoch 10 Batch 300 Loss 0.8440757989883423
Epoch 10 Batch 400 Loss 0.8192751407623291
Epoch 10 Batch 500 Loss 0.8481991291046143
Epoch 10 Batch 600 Loss 0.9068182110786438
Epoch 10 Batch 700 Loss 0.8095198273658752
Epoch 10 Batch 800 Loss 0.7869399785995483
Epoch 10 Batch 900 Loss 0.8212043642997742
Epoch 10 Loss 0.8231
Time taken for 1 epoch 67.14899373054504 sec

트레이닝이 끝났습니다!
```

## 트레이닝이 끝난 모델을 이용한 샘플링

```python
sampling_RNN_model = RNN(batch_size=1)
sampling_RNN_model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
sampling_RNN_model.build(tf.TensorShape([1, None]))
sampling_RNN_model.summary()

# 샘플링을 시작합니다.
print("샘플링을 시작합니다!")
print(generate_text(sampling_RNN_model, start_string=u' '))
```

```
Model: "rnn_3"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
embedding_3 (Embedding)      multiple                  25344     
_________________________________________________________________
lstm_3 (LSTM)                multiple                  5246976   
_________________________________________________________________
dense_3 (Dense)              multiple                  101475    
=================================================================
Total params: 5,373,795
Trainable params: 5,373,795
Non-trainable params: 0
_________________________________________________________________
샘플링을 시작합니다!
 ss           printf(" .. MAX_LOCK_USASHIC_PRINT_TASK_TIMEKEEntry))
		if (diag) {
		reset_free_percpu(skb);
		again_ops->proc_cap_buffer = bytesn;
	}

	/*
	 * Determine if CONFIG_SCHEDSTATS */

	/* Keep start */
	dr->flags &= ~SPONT_PAREN; i++) {
		kdb_printf("kdb enabled" by this code trigger that may have been actually. If all
 * threads can possibly get_kprojid to insert a CPU with swsusp_bit)
		.binary);
	}
	kdb_register_flags(_m);
	__add_notify(dest);
}

static DEFINE_MUTEX(entry_count" },
	{ CTL_INT,	FS_REAL,		"pall" },
	/* KERN_s reschedule if verified to etc. Watchdog namespace
	 * data is not yet, disable installs we need multiple.  Note:
	 * See prepare_creds() values in a sock when might be
	 * been the timexw->goto out;
	r_start and characters. */
static int show_rcubarrierdev(struct request_queue *q, struct rw_semaphore *seturn -EINVAL one reads */
	if (isdigit(TASKLET_STATE_COMING &&
	    state == KDB_CMD_CPU) {
		if (atomic_inc_retitid == CONFIG_MODULE_NO_BRL_0) {
		kdb_printf("%s: at syscall audit message or not all of system callbacks while shorted event
		 * instruction within we check the
			 * callback is the offset of arch pages were above.
			 */
			if (nextarg > name)
				len += kprobe_dit(nlst, userstromph_write,
				&mod->mkobE_SHIFT,
			rdp->nocb_head);
	up_read(&mm->mmareeder.
 */
static void blk_add_trace_rq_insert(struct ring_buffer_percpu *cpu_buffer = buffer->buffers;
	struct ring_buffer_per_cpu *cpu_buffer;
	unsigned long pfn = res->flags;

	/*
	 * Unbind are files. The safe if there is nothing towards the list, it has already
	 * stop may canclude <linux/kmod.h>
#include <linux/shareded.h>
#include <linux/proc_fs.h>
#include <linux/fs.h>
#include <linux/pidfcer.h>
#include <linux/binfmtx].mutex);
}
EXPORT_SYMBOL(add_taint);

static void kallsyms_lookup(unsigned long ip, unsigned long *;

	struct seq_file *mit;
	int ret;

	if (!file)
		options = true;
}
/*
 * get_parent - Free a given event that,
 * second for all kthread_cpu - atomically allocate and modify
 * 8, 1916 -  success - crc = -1, ns
 * an option is placed, it out of them symbol and parsing entries with
 *	trace-issues.
 * @line: The image hierarchies on correctned by Nick Joss, ArjaN wereo);

	if (nr_late != header && flags || !pre_mask)
		goto error;

	bt->nr_cpu_ids; i++) {
		static int run_read_unlock_sched_timer(struct ptrace_remove_work aterator to start checkevent_utr_task(struct rt_mutex *lock)
{
	return __alarm_base_files(old, callchain_must_stric, current);
}

user_resource(r, p);
		else
			reset_iter(&bpage->elements);
		bcfs_validate_change(char *cmdline,
				     unsigned long address,
				unsigned long min_sze,
			        char	*bufptr, is_read,
		struct blk_irq_bw *blk_add_trace_rq, int permission== buf_addr)
{
	struct module_kobject *mk;
	int cpu;

	pr_devel("<%s\n", __enterisers - ring buffer's process or already confindings.h"
 "write: {
	LOGG_COUNT;
	pr_info("\t%d %ld", pid_q bin[i], \
				     next_page, *respage), PAGE_SIZE))
			avail = strchr(sp, right);
		bit(val, &val), event) defined(CONFIG_MMU
	{
		struct task_numa_free directory parsed;
			if (!s->ss->cfs_cape.start_lba
		redister_sysctl();

		switch (c->type) {
			if (ressize(&right = frozen - low_fetch_irq)(SU_DESTREAD_ALLOW, &root_desc);
		if (rdtp->dynticks_idle_nestint	= 0;
								    RLIME_NFP fmt;
	arm_timer(timr->it_i_uprobe_buffer, 1);
			elta trace_add_unbid
		S_INTMASK,		"ip_opport
		 * disarm entries:
			*/
			if (argc != 1
	    && oms[0]) {
			if (!te_cpu_base->cpumask.cbcpu)
				break;
		}
		acct_acquire(nval, &val);
		return TRACE_TYPE_HASH_BITS;
	}

	/* Check synchronize installs, but we assume missing a trace probe */
	proc_watchdog();
	kthread to the new kernel process.
 *
 * @start: start address
 * @arg:	argv[3]
	module_usecsize 16:
		disable_irq = 0;
	} while (read_seqcount_ble projes starts bA.
	 */
	copied = true;
	else if ((strctx->flags & KEXEC_FILE) &&
	      elemenable)
			alarm->node = attr->addr;
		break;
	case Audit_block from us
```