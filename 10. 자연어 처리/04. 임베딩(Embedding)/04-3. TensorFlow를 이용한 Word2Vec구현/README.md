## 라이브러리 import

```python
import io
import re
import string
import tensorflow as tf
import tqdm

from tensorflow.keras import Model
from tensorflow.keras.layers import Dot, Embedding, Flatten
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
```

## TensorBoard 로드

```python
# Load the TensorBoard notebook extension
%load_ext tensorboard
```

```python
SEED = 42
AUTOTUNE = tf.data.AUTOTUNE
```

## 단어 Tokenizing

```python
sentence = "The wide road shimmered in the hot sun"
tokens = list(sentence.lower().split())
print(len(tokens))
print(tokens)
```

```python
8
['the', 'wide', 'road', 'shimmered', 'in', 'the', 'hot', 'sun']
```

## 단어별 id 부여 및 vocab dictionary 생성

```python
vocab, index = {}, 1  # start indexing from 1
vocab['<pad>'] = 0  # add a padding token
for token in tokens:
  if token not in vocab:
    vocab[token] = index
    index += 1
vocab_size = len(vocab)
print(vocab)
```

```python
{'<pad>': 0, 'the': 1, 'wide': 2, 'road': 3, 'shimmered': 4, 'in': 5, 'hot': 6, 'sun': 7}
```

```python
example_sequence = [vocab[word] for word in tokens]
print(example_sequence)
```

```
[1, 2, 3, 4, 5, 1, 6, 7]
```

## Skip-gram 모델을 위한 target, context 쌍 생성

```python
print(sentence)
```

```
The wide road shimmered in the hot sun
```

```python
window_size = 2
positive_skip_grams, _ = tf.keras.preprocessing.sequence.skipgrams(
      example_sequence,
      vocabulary_size=vocab_size,
      window_size=window_size,
      negative_samples=0)
print(len(positive_skip_grams))
```

```
26
```

```python
for target, context in positive_skip_grams[:5]:
  print(f"({target}, {context}): ({inverse_vocab[target]}, {inverse_vocab[context]})")
```

```
(4, 3): (shimmered, road)
(7, 1): (sun, the)
(4, 1): (shimmered, the)
(2, 4): (wide, shimmered)
(4, 2): (shimmered, wide)
```

## 학습을 위한 Negative Sampling

```python
# Get target and context words for one positive skip-gram.
target_word, context_word = positive_skip_grams[0]
print(inverse_vocab[context_word])

# Set the number of negative samples per positive context.
# positive example 1개당 4개의 negative example을 부여합니다.
num_ns = 4

context_class = tf.reshape(tf.constant(context_word, dtype="int64"), (1, 1))

negative_sampling_candidates, _, _ = tf.random.log_uniform_candidate_sampler(
    true_classes=context_class,  # class that should be sampled as 'positive'
    num_true=1,  # each positive skip-gram has 1 positive context class
    num_sampled=num_ns,  # number of negative context words to sample
    unique=True,  # all the negative samples should be unique
    range_max=vocab_size,  # pick index of the samples from [0, vocab_size]
    seed=SEED,  # seed for reproducibility
    name="negative_sampling"  # name of this operation
)
print(negative_sampling_candidates)
print([inverse_vocab[index.numpy()] for index in negative_sampling_candidates])
```

```
road
tf.Tensor([2 1 4 3], shape=(4,), dtype=int64)
['wide', 'the', 'shimmered', 'road']
```

## Negative Example을 포함한 학습 데이터셋을 구성합니다.

```python
# Add a dimension so you can use concatenation (on the next step).
negative_sampling_candidates = tf.expand_dims(negative_sampling_candidates, 1)

# Concat positive context word with negative sampled words.
context = tf.concat([context_class, negative_sampling_candidates], 0)

# Label first context word as 1 (positive) followed by num_ns 0s (negative).
label = tf.constant([1] + [0]*num_ns, dtype="int64")

# Reshape target to shape (1,) and context and label to (num_ns+1,).
target = tf.squeeze(target_word)
context = tf.squeeze(context)
label = tf.squeeze(label)
```

```python
print(f"target_index    : {target}")
print(f"target_word     : {inverse_vocab[target_word]}")
print(f"context_indices : {context}")
print(f"context_words   : {[inverse_vocab[c.numpy()] for c in context]}")
print(f"label           : {label}")
```

```
target_index    : 4
target_word     : shimmered
context_indices : [3 2 1 4 3]
context_words   : ['road', 'wide', 'the', 'shimmered', 'road']
label           : [1 0 0 0 0]
```

```python
print("target  :", target)
print("context :", context)
print("label   :", label)
```

```
target  : tf.Tensor(4, shape=(), dtype=int32)
context : tf.Tensor([3 2 1 4 3], shape=(5,), dtype=int64)
label   : tf.Tensor([1 0 0 0 0], shape=(5,), dtype=int64)
```

## 어휘(Vocabulary)별 가중치 적용하기

- 'the','is','on' 같은 단어들은 출현빈도는 높지만 새로운 정보는 많이 제공하지 못합니다. 따라서 단어별 가중치 정도를 적용해줍니다

```python
sampling_table = tf.keras.preprocessing.sequence.make_sampling_table(size=10)
print(sampling_table)
```

```
[0.00315225 0.00315225 0.00547597 0.00741556 0.00912817 0.01068435
 0.01212381 0.01347162 0.01474487 0.0159558 ]
```

```python
# Generates skip-gram pairs with negative sampling for a list of sequences
# (int-encoded sentences) based on window size, number of negative samples
# and vocabulary size.
def generate_training_data(sequences, window_size, num_ns, vocab_size, seed):
  # Elements of each training example are appended to these lists.
  targets, contexts, labels = [], [], []

  # Build the sampling table for vocab_size tokens.
  sampling_table = tf.keras.preprocessing.sequence.make_sampling_table(vocab_size)

  # Iterate over all sequences (sentences) in dataset.
  for sequence in tqdm.tqdm(sequences):

    # Generate positive skip-gram pairs for a sequence (sentence).
    positive_skip_grams, _ = tf.keras.preprocessing.sequence.skipgrams(
          sequence,
          vocabulary_size=vocab_size,
          sampling_table=sampling_table,
          window_size=window_size,
          negative_samples=0)

    # Iterate over each positive skip-gram pair to produce training examples
    # with positive context word and negative samples.
    for target_word, context_word in positive_skip_grams:
      context_class = tf.expand_dims(
          tf.constant([context_word], dtype="int64"), 1)
      negative_sampling_candidates, _, _ = tf.random.log_uniform_candidate_sampler(
          true_classes=context_class,
          num_true=1,
          num_sampled=num_ns,
          unique=True,
          range_max=vocab_size,
          seed=SEED,
          name="negative_sampling")

      # Build context and label vectors (for one target word)
      negative_sampling_candidates = tf.expand_dims(
          negative_sampling_candidates, 1)

      context = tf.concat([context_class, negative_sampling_candidates], 0)
      label = tf.constant([1] + [0]*num_ns, dtype="int64")

      # Append each element from the training example to global lists.
      targets.append(target_word)
      contexts.append(context)
      labels.append(label)

  return targets, contexts, labels
```

## 학습을 위한 텍스트 파일을 준비합니다.

```python
path_to_file = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')
```

```
Downloading data from https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt
1122304/1115394 [==============================] - 0s 0us/step
1130496/1115394 [==============================] - 0s 0us/step
```

```python
with open(path_to_file) as f:
  lines = f.read().splitlines()
for line in lines[:20]:
  print(line)
```

```
First Citizen:
Before we proceed any further, hear me speak.

All:
Speak, speak.

First Citizen:
You are all resolved rather to die than to famish?

All:
Resolved. resolved.

First Citizen:
First, you know Caius Marcius is chief enemy to the people.

All:
We know't, we know't.

First Citizen:
Let us kill him, and we'll have corn at our own price.
```

## 비어 있는 줄을 제거합니다.

```python
text_ds = tf.data.TextLineDataset(path_to_file).filter(lambda x: tf.cast(tf.strings.length(x), bool))
```

## 단어들을 벡터화합니다.

```python
# Now, create a custom standardization function to lowercase the text and
# remove punctuation.
def custom_standardization(input_data):
  lowercase = tf.strings.lower(input_data)
  return tf.strings.regex_replace(lowercase,
                                  '[%s]' % re.escape(string.punctuation), '')


# Define the vocabulary size and number of words in a sequence.
vocab_size = 4096
sequence_length = 10

# Use the text vectorization layer to normalize, split, and map strings to
# integers. Set output_sequence_length length to pad all samples to same length.
vectorize_layer = TextVectorization(
    standardize=custom_standardization,
    max_tokens=vocab_size,
    output_mode='int',
    output_sequence_length=sequence_length)
```

```python
vectorize_layer.adapt(text_ds.batch(1024))
```

## 출력 순서는 빈도에 기반한 내림차순입니다.

```python
# Save the created vocabulary for reference.
inverse_vocab = vectorize_layer.get_vocabulary()
print(inverse_vocab[:20])
```

```
['', '[UNK]', 'the', 'and', 'to', 'i', 'of', 'you', 'my', 'a', 'that', 'in', 'is', 'not', 'for', 'with', 'me', 'it', 'be', 'your']
```

```python
# Vectorize the data in text_ds.
text_vector_ds = text_ds.batch(1024).prefetch(AUTOTUNE).map(vectorize_layer).unbatch()
```

```python
sequences = list(text_vector_ds.as_numpy_iterator())
print(len(sequences))
```

```
32777
```

## 준비한 데이터의 일부를 살펴봅니다.

```python
for seq in sequences[:5]:
  print(f"{seq} => {[inverse_vocab[i] for i in seq]}")
```

```
[ 89 270   0   0   0   0   0   0   0   0] => ['first', 'citizen', '', '', '', '', '', '', '', '']
[138  36 982 144 673 125  16 106   0   0] => ['before', 'we', 'proceed', 'any', 'further', 'hear', 'me', 'speak', '', '']
[34  0  0  0  0  0  0  0  0  0] => ['all', '', '', '', '', '', '', '', '', '']
[106 106   0   0   0   0   0   0   0   0] => ['speak', 'speak', '', '', '', '', '', '', '', '']
[ 89 270   0   0   0   0   0   0   0   0] => ['first', 'citizen', '', '', '', '', '', '', '', '']
```

## 트레이닝 데이터를 생성합니다.

```python
targets, contexts, labels = generate_training_data(
    sequences=sequences,
    window_size=2,
    num_ns=4,
    vocab_size=vocab_size,
    seed=SEED)
print(len(targets), len(contexts), len(labels))
```

```
100%|██████████| 32777/32777 [00:09<00:00, 3318.04it/s]65586 65586 65586
```

## batch 단위로 데이터셋을 묶습니다.

- ((target_word, context_word), (label)) 형태로 데이터가 구성되게됩니다.

```python
BATCH_SIZE = 1024
BUFFER_SIZE = 10000
dataset = tf.data.Dataset.from_tensor_slices(((targets, contexts), labels))
dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
print(dataset)
```

```python
<BatchDataset shapes: (((1024,), (1024, 5, 1)), (1024, 5)), types: ((tf.int32, tf.int64), tf.int64)>
```

```python
dataset = dataset.cache().prefetch(buffer_size=AUTOTUNE)
print(dataset)
```

```
<PrefetchDataset shapes: (((1024,), (1024, 5, 1)), (1024, 5)), types: ((tf.int32, tf.int64), tf.int64)>
```

## Word2Vec 모델을 정의합니다.

```python
class Word2Vec(Model):
  def __init__(self, vocab_size, embedding_dim):
    super(Word2Vec, self).__init__()
    self.target_embedding = Embedding(vocab_size,
                                      embedding_dim,
                                      input_length=1,
                                      name="w2v_embedding")
    self.context_embedding = Embedding(vocab_size,
                                       embedding_dim,
                                       input_length=num_ns+1)
    self.dots = Dot(axes=(3, 2))
    self.flatten = Flatten()

  def call(self, pair):
    target, context = pair
    word_emb = self.target_embedding(target)
    context_emb = self.context_embedding(context)
    dots = self.dots([context_emb, word_emb])
    return self.flatten(dots)
```

## compile 함수로 학습에 필요한 설정을 지정합니다.

```python
embedding_dim = 128
word2vec = Word2Vec(vocab_size, embedding_dim)
word2vec.compile(optimizer='adam',
                 loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                 metrics=['accuracy'])
```

## TensorBoard 로그를 지정합니다.

```python
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="logs")
```

```python
word2vec.fit(dataset, epochs=20, callbacks=[tensorboard_callback])
```

```
Epoch 1/20
64/64 [==============================] - 2s 20ms/step - loss: 1.6081 - accuracy: 0.2326
Epoch 2/20
64/64 [==============================] - 1s 16ms/step - loss: 1.5875 - accuracy: 0.5498
Epoch 3/20
64/64 [==============================] - 1s 16ms/step - loss: 1.5367 - accuracy: 0.5826
Epoch 4/20
64/64 [==============================] - 1s 16ms/step - loss: 1.4528 - accuracy: 0.5576
Epoch 5/20
64/64 [==============================] - 1s 16ms/step - loss: 1.3559 - accuracy: 0.5700
Epoch 6/20
64/64 [==============================] - 1s 16ms/step - loss: 1.2609 - accuracy: 0.5992
Epoch 7/20
64/64 [==============================] - 1s 16ms/step - loss: 1.1722 - accuracy: 0.6342
Epoch 8/20
64/64 [==============================] - 1s 16ms/step - loss: 1.0896 - accuracy: 0.6696
Epoch 9/20
64/64 [==============================] - 1s 16ms/step - loss: 1.0128 - accuracy: 0.7028
Epoch 10/20
64/64 [==============================] - 1s 16ms/step - loss: 0.9413 - accuracy: 0.7325
Epoch 11/20
64/64 [==============================] - 1s 15ms/step - loss: 0.8748 - accuracy: 0.7587
Epoch 12/20
64/64 [==============================] - 1s 15ms/step - loss: 0.8133 - accuracy: 0.7823
Epoch 13/20
64/64 [==============================] - 1s 16ms/step - loss: 0.7566 - accuracy: 0.8017
Epoch 14/20
64/64 [==============================] - 1s 16ms/step - loss: 0.7044 - accuracy: 0.8186
Epoch 15/20
64/64 [==============================] - 1s 16ms/step - loss: 0.6565 - accuracy: 0.8343
Epoch 16/20
64/64 [==============================] - 1s 16ms/step - loss: 0.6128 - accuracy: 0.8484
Epoch 17/20
64/64 [==============================] - 1s 17ms/step - loss: 0.5728 - accuracy: 0.8613
Epoch 18/20
64/64 [==============================] - 1s 16ms/step - loss: 0.5364 - accuracy: 0.8726
Epoch 19/20
64/64 [==============================] - 1s 16ms/step - loss: 0.5032 - accuracy: 0.8824
Epoch 20/20
64/64 [==============================] - 1s 15ms/step - loss: 0.4730 - accuracy: 0.8916
<keras.callbacks.History at 0x7f80d7740710>
```

## 학습에 대한 TensorBoard 로그를 살펴봅니다.

```python
#docs_infra: no_execute
%tensorboard --logdir logs
```

## Embedding layer의 가중치와 vocab 정보를 추출합니다.

```python
weights = word2vec.get_layer('w2v_embedding').get_weights()[0]
vocab = vectorize_layer.get_vocabulary()
```

```python
print(weights)
print(weights[0].shape)
print(weights.shape)
print(vocab)
```

```
[-0.0474758  -0.01889622 -0.00060091 ...  0.01868533  0.03912387
  -0.0073231 ]
 [-0.05322375  0.29298022 -0.1170481  ...  0.02917635  0.3245909
   0.18440294]
 [ 0.21415842  0.00582969 -0.02878693 ... -0.10956954  0.30038247
   0.102309  ]
 ...
 [-0.25058842  0.07132742 -0.08693206 ... -0.20305537 -0.06410046
  -0.04000486]
 [-0.05651178  0.05713945  0.10277442 ...  0.11937515 -0.04790678
   0.08634761]
 [-0.17136832 -0.04823127  0.1196787  ...  0.17893475 -0.11609387
   0.06819272]]
(128,)
(4096, 128)
['', '[UNK]', 'the', 'and', 'to', 'i', 'of', 'you', 'my', 'a', 'that', 'in', 'is', 'not', 'for', 'with', 'me', 'it', 'be', 'your', 'his', 'this', 'but', 'he', 'have', 'as', 'thou', 'him', 'so', 'what', 'thy', 'will', 'no', 'by', 'all', 'king', 'we', 'shall', 'her', 'if', 'our', 'are', 'do', 'thee', 'now', 'lord', 'good', 'on', 'o', 'come', 'from', 'sir', 'or', 'which', 'more', 'then', 'well', 'at', 'would', 'was', 'they', 'how', 'here', 'she', 'than', 'their', 'them', 'ill', 'duke', 'am', 'hath', 'say', 'let', 'when', 'one', 'go', 'were', 'love', 'may', 'us', 'make', 'upon', 'yet', 'richard', 'like', 'there', 'must', 'should', 'an', 'first', 'why', 'queen', 'had', 'know', 'man', 'did', 'tis', 'where', 'see', 'some', 'too', 'death', 'give', 'who', 'these', 'take', 'speak', 'edward', 'york', 'mine', 'such', 'up', 'out', 'henry', 'romeo', 'can', 'father', 'tell', 'time', 'gloucester', 'most', 'lady', 'son', 'nor', 'vincentio', 'hear', 'life', 'god', 'made', 'art', 'warwick', 'think', 'much', 'heart', 'never', 'doth', 'brother', 'ay', 'before', 'true', 'both', 'thus', 'cannot', 'petruchio', 'any', 'being', 'away', 'blood', 'name', 'fair', 'coriolanus', 'been', 'noble', 'men', 'menenius', 'look', 'again', 'very', 'hand', 'day', 'pray', 'own', 'juliet', 'done', 'sweet', 'second', 'myself', 'therefore', 'leave', 'great', 'against', 'though', 'poor', 'honour', 'down', 'prince', 'hast', 'way', 'angelo', 'fear', 'old', 'nay', 'heaven', 'clarence', 'till', 'call', 'eyes', 'world', 'stay', 'live', 'stand', 'nurse', 'grace', 'many', 'comes', 'ever', 'even', 'wife', 'nothing', 'iii', 'die', 'dead', 'whose', 'bear', 'night', 'other', 'isabella', 'bolingbroke', 'friends', 'leontes', 'head', 'friar', 'peace', 'buckingham', 'unto', 'those', 'better', 'lords', 'word', 'off', 'gone', 'tranio', 'two', 'mother', 'hence', 'marcius', 'house', 'still', 'since', 'news', 'lucio', 'could', 'sicinius', 'master', 'little', 'gentleman', 'daughter', 'soul', 'thing', 'put', 'once', 'whom', 'margaret', 'set', 'long', 'himself', 'face', 'camillo', 'words', 'thine', 'iv', 'ere', 'else', 'elizabeth', 'capulet', 'none', 'katharina', 'rest', 'might', 'madam', 'gods', 'crown', 'best', 'young', 'power', 'pardon', 'tongue', 'dear', 'part', 'farewell', 'citizen', 'vi', 'bring', 'keep', 'ii', 'hope', 'gentle', 'right', 'hastings', 'said', 'lucentio', 'find', 'every', 'hortensio', 'forth', 'came', 'bid', 'home', 'hands', 'earth', 'welcome', 'lets', 'into', 'brutus', 'hold', 'dost', 'cause', 'tears', 'boy', 'baptista', 'back', 'about', 'rome', 'please', 'thats', 'provost', 'mistress', 'people', 'makes', 'help', 'indeed', 'heard', 'full', 'cousin', 'thought', 'show', 'after', 'wilt', 'place', 'mind', 'has', 'grumio', 'escalus', 'cominius', 'pompey', 'marry', 'husband', 'whats', 'friend', 'state', 'shame', 'mean', 'within', 'gremio', 'while', 'servant', 'clifford', 'only', 'hither', 'hes', 'fathers', 'gracious', 'aufidius', 'royal', 'rather', 'prove', 'northumberland', 'last', 'far', 'eye', 'duchess', 'answer', 'thousand', 'mercutio', 'lie', 'third', 'another', 'paulina', 'meet', 'claudio', 'murderer', 'shalt', 'lay', 'child', 'use', 'sorrow', 'lies', 'joy', 'comfort', 'beseech', 'tomorrow', 'grief', 'benvolio', 'war', 'sun', 'polixenes', 'less', 'kings', 'autolycus', 'years', 'hour', 'happy', 'fortune', 'fellow', 'end', 'thank', 'montague', 'holy', 'business', 'things', 'matter', 'hate', 'grave', 'faith', 'enough', 'arms', 'truth', 'prospero', 'lives', 'light', 'clown', 'bed', 'without', 'three', 'save', 'pity', 'fall', 'body', 'nature', 'kind', 'get', 'tybalt', 'turn', 'theres', 'swear', 'send', 'laurence', 'kate', 'days', 'saw', 'hell', 'fight', 'false', 'bianca', 'believe', 'ah', 'law', 'follow', 'yourself', 'yours', 'villain', 'uncle', 'present', 'means', 'looks', 'does', 'volumnia', 'talk', 'sleep', 'neer', 'left', 'foul', 'bloody', 'aumerle', 'anne', 'city', 'worthy', 'woman', 'told', 'servingman', 'proud', 'need', 'mercy', 'land', 'hearts', 'brothers', 'wish', 'sound', 'sit', 'shepherd', 'sea', 'play', 'messenger', 'together', 'thyself', 'sword', 'justice', 'high', 'gaunt', 'doubt', 'breath', 'because', 'woe', 'through', 'majesty', 'john', 'horse', 'fire', 'either', 'biondello', 'under', 'sovereign', 'sister', 'seen', 'paris', 'heres', 'children', 'catesby', 'break', 'maid', 'heavy', 'twas', 'times', 'signior', 'lost', 'gentlemen', 'content', 'care', 'wrong', 'traitor', 'says', 'ready', 'purpose', 'oath', 'norfolk', 'loves', 'kill', 'haste', 'fly', 'florizel', 'edwards', 'ears', 'canst', 'alas', 'age', 'weep', 'warrant', 'strange', 'reason', 'new', 'liege', 'late', 'hermione', 'grey', 'each', 'alone', 'thoughts', 'thanks', 'stands', 'slain', 'sent', 'rivers', 'remember', 'kiss', 'hours', 'cry', 'brought', 'strike', 'straight', 'spirit', 'seem', 'return', 'married', 'mark', 'loss', 'lest', 'knows', 'having', 'ground', 'free', 'france', 'didst', 'charge', 'souls', 'service', 'sebastian', 'report', 'person', 'near', 'found', 'cold', 'born', 'yea', 'ye', 'widow', 'sin', 'sight', 'senator', 'richmond', 'lose', 'heavens', 'desire', 'common', 'yield', 'twere', 'tender', 'sure', 'soon', 'shes', 'itself', 'general', 'foot', 'fool', 'fie', 'air', 'youll', 'women', 'saint', 'prithee', 'prison', 'ha', 'fault', 'enemy', 'duty', 'deep', 'citizens', 'arm', 'unless', 'twenty', 'trust', 'sons', 'patience', 'past', 'mad', 'known', 'honest', 'deed', 'command', 'antonio', 'yes', 'ten', 'something', 'oxford', 'oer', 'lewis', 'heir', 'fearful', 'dare', 'cut', 'bound', 'withal', 'wear', 'voices', 'sirrah', 'serve', 'revenge', 'point', 'open', 'masters', 'highness', 'half', 'gonzalo', 'ear', 'draw', 'company', 'beauty', 'beat', 'worse', 'virtue', 'tale', 'read', 'next', 'neither', 'marriage', 'ist', 'hark', 'brave', 'youth', 'work', 'side', 'seek', 'sake', 'pleasure', 'methinks', 'loving', 'lancaster', 'knew', 'field', 'enemies', 'country', 'counsel', 'change', 'become', 'ask', 'wars', 'tower', 'today', 'soldiers', 'pale', 'miranda', 'met', 'given', 'gates', 'further', 'fast', 'court', 'coming', 'behold', 'took', 'tonight', 'themselves', 'strength', 'run', 'rich', 'peter', 'merry', 'lips', 'ho', 'gave', 'elbow', 'course', 'confess', 'tribunes', 'title', 'sworn', 'same', 'sad', 'pass', 'just', 'curse', 'between', 'banishd', 'already', 'along', 'wind', 'whilst', 'thomas', 'spoke', 'post', 'pluck', 'perdita', 'office', 'loved', 'issue', 'goes', 'george', 'england', 'dream', 'bosom', 'bold', 'bad', 'wounds', 'subject', 'stanley', 'over', 'officer', 'learn', 'jest', 'hadst', 'fetch', 'thence', 'suit', 'speed', 'soldier', 'sly', 'presence', 'often', 'green', 'grant', 'forget', 'foe', 'fit', 'faults', 'entreat', 'dangerous', 'worth', 'wit', 'wert', 'wast', 'voice', 'throne', 'strong', 'soft', 'seems', 'request', 'rage', 'quarrel', 'prayers', 'order', 'mighty', 'mariana', 'heads', 'gold', 'deny', 'black', 'attend', 'win', 'wherein', 'want', 'throw', 'thither', 'somerset', 'princely', 'morning', 'march', 'mans', 'lookd', 'london', 'lartius', 'devil', 'deeds', 'consent', 'case', 'calld', 'buy', 'begin', 'battle', 'appear', 'almost', 'wonder', 'wise', 'whence', 'water', 'watch', 'valiant', 'subjects', 'speech', 'ratcliff', 'pretty', 'presently', 'others', 'music', 'move', 'killd', 'hot', 'honours', 'harm', 'got', 'feel', 'earl', 'consul', 'brief', 'breast', 'ancient', 'above', 'whether', 'towards', 'tot', 't', 'slave', 'roman', 'poison', 'murder', 'mowbray', 'measure', 'mayor', 'least', 'knock', 'held', 'hang', 'forward', 'five', 'eer', 'alack', '3', 'wouldst', 'witness', 'virgilia', 'touch', 'promise', 'mortal', 'moon', 'living', 'keeper', 'hereford', 'hearing', 'flesh', 'em', 'curtis', 'close', 'awhile', 'ariel', 'worst', 'wherefore', 'weeping', 'virtuous', 'toward', 'quickly', 'pay', 'patient', 'ourselves', 'letters', 'lawful', 'guilty', 'grow', 'greater', 'goodly', 'dot', 'deserved', 'chance', 'breathe', 'blows', 'besides', 'base', 'antigonus', 'anon', 'worship', 'woes', 'whither', 'truly', 'tear', 'stir', 'small', 'shows', 'pedant', 'ont', 'offence', 'note', 'needs', 'mouth', 'longer', 'ladies', 'henrys', 'golden', 'fled', 'feast', 'exeter', 'englands', 'derby', 'daughters', 'beg', 'banishment', 'angry', 'year', 'won', 'until', 'traitors', 'town', 'taen', 'sudden', 'speaks', 'slew', 'short', 'shed', 'sense', 'rough', 'queens', 'plain', 'padua', 'liberty', 'lands', 'laid', 'knee', 'isabel', 'hard', 'enter', 'doing', 'defend', 'clouds', 'choose', 'book', 'bohemia', 'behind', 'bears', 'aside', 'amen', 'action', 'vow', 'visit', 'victory', 'stood', 'prisoner', 'princes', 'plantagenet', 'piece', 'oft', 'noise', 'neck', 'mothers', 'lived', 'leisure', 'lead', 'kingdom', 'int', 'hundred', 'hide', 'four', 'fine', 'dry', 'deliver', 'conscience', 'commend', 'certain', 'broke', 'bride', 'bones', 'blow', 'bawd', 'act', 'yourselves', 'xi', 'womb', 'wheres', 'watchman', 'wash', 'twixt', 'twice', 'teach', 'shake', 'seat', 'rutland', 'rise', 'proceed', 'pride', 'morrow', 'letter', 'knowledge', 'knees', 'kept', 'herself', 'glory', 'glad', 'fortunes', 'forgot', 'forbid', 'few', 'drink', 'danger', 'countrys', 'chamber', 'bushy', 'blessed', 'bitter', 'barnardine', 'alive', 'adieu', 'weak', 'walk', 'vain', 'tongues', 'thereof', 'tent', 'stars', 'sick', 'secret', 'sampson', 'remain', 'quoth', 'quiet', 'promised', 'prepare', 'page', 'moved', 'lordship', 'knowst', 'honourable', 'guess', 'going', 'gainst', 'fresh', 'fought', 'flowers', 'dorset', 'dog', 'dispatch', 'died', 'despair', 'deadly', 'cheeks', 'calls', 'brakenbury', 'bless', 'becomes', 'banished', 'wretch', 'wood', 'woo', 'wisdom', 'white', 'whiles', 'tyrrel', 'try', 'suffer', 'struck', 'stop', 'steel', 'spent', 'sort', 'smile', 'shore', 'seven', 'precious', 'pain', 'ours', 'offer', 'obey', 'musician', 'money', 'mens', 'making', 'liest', 'lack', 'labour', 'kneel', 'kinsman', 'kindness', 'humble', 'gregory', 'gives', 'girl', 'froth', 'form', 'force', 'foes', 'fellows', 'fare', 'english', 'early', 'dreams', 'doom', 'deaths', 'cried', 'cheer', 'charity', 'cast', 'army', 'weary', 'warm', 'walls', 'volsces', 'vengeance', 'trial', 'treason', 'thinks', 'therein', 'swords', 'six', 'silence', 'scorn', 'satisfied', 'safe', 'repent', 'quick', 'mayst', 'match', 'manner', 'malice', 'judge', 'its', 'idle', 'ghost', 'frown', 'foolish', 'fond', 'flatter', 'encounter', 'eat', 'desperate', 'dark', 'dance', 'crave', 'church', 'carry', 'bishop', 'birth', 'bids', 'abroad', 'writ', 'vile', 'valour', 'valeria', 'tut', 'trouble', 'touchd', 'taken', 'tailor', 'stone', 'spake', 'sour', 'sometime', 'sentence', 'sacred', 'received', 'reasons', 'quite', 'party', 'overdone', 'nine', 'mock', 'mistake', 'meat', 'meaning', 'maids', 'lo', 'lend', 'honesty', 'health', 'harry', 'grown', 'fort', 'fell', 'favour', 'fain', 'execution', 'excuse', 'evil', 'dukes', 'drunk', 'drawn', 'divine', 'deserve', 'depart', 'cruel', 'county', 'corse', 'corioli', 'changed', 'caius', 'bona', 'blunt', 'blame', 'bite', 'betwixt', 'belike', 'awake', 'among', 'always', 'aid', 'affection', 'account', 'wrongs', 'wretched', 'whos', 'whereof', 'went', 'wench', 'vice', 'understand', 'trumpets', 'swift', 'steal', 'sits', 'showd', 'self', 'scarce', 'safety', 'remedy', 'reign', 'receive', 'percy', 'pains', 'ones', 'occasion', 'names', 'marshal', 'mantua', 'leaves', 'kindred', 'judgment', 'ild', 'humour', 'hair', 'guard', 'gown', 'flower', 'effect', 'easy', 'dust', 'dull', 'drop', 'door', 'deputy', 'deceived', 'cursed', 'cross', 'coward', 'condition', 'choice', 'castle', 'captain', 'burn', 'buried', 'broken', 'bright', 'blind', 'blest', 'although', 'abhorson', 'write', 'womans', 'wings', 'wild', 'wicked', 'weeds', 'wall', 'wail', 'view', 'used', 'undone', 'tyrant', 'troth', 'triumph', 'tidings', 'thinkst', 'taste', 'spirits', 'sorry', 'sorrows', 'sing', 'simple', 'silver', 'senators', 'seize', 'seest', 'seal', 'rotten', 'resolved', 'resolve', 'quit', 'proclaim', 'private', 'priest', 'practise', 'pleased', 'plague', 'perceive', 'oracle', 'nose', 'mopsa', 'mamillius', 'laugh', 'ladys', 'keeps', 'ignorant', 'hie', 'fury', 'former', 'fools', 'fill', 'fatal', 'fares', 'envy', 'ease', 'despite', 'deposed', 'cure', 'creature', 'courage', 'cares', 'capitol', 'brows', 'brow', 'bend', 'authority', 'absence', 'yonder', 'woful', 'wives', 'wine', 'whole', 'wanton', 'vantage', 'twelve', 'turns', 'tune', 'tread', 'thursday', 'throat', 'thrive', 'teeth', 'tedious', 'summer', 'stones', 'sport', 'spare', 'sometimes', 'slander', 'sign', 'sighs', 'shouldst', 'ship', 'seeing', 'rude', 'ross', 'rid', 'respect', 'question', 'proof', 'prepared', 'possible', 'pisa', 'perhaps', 'passing', 'parts', 'owe', 'nights', 'list', 'knave', 'intend', 'husbands', 'humbly', 'hit', 'hanging', 'gentlewoman', 'garments', 'gain', 'follows', 'followers', 'fiery', 'farther', 'envious', 'due', 'dreadful', 'double', 'devise', 'darest', 'cunning', 'crowns', 'cries', 'countenance', 'corn', 'claim', 'chide', 'cheek', 'cell', 'borne', 'benefit', 'begins', 'beggar', 'bark', 'banish', 'aught', 'asleep', 'anger', 'affairs', 'advise', 'wrath', 'wound', 'wot', 'wont', 'wits', 'withdraw', 'warwicks', 'violent', 'verona', 'untimely', 'undertake', 'success', 'storm', 'spring', 'slaughterd', 'sins', 'shadow', 'several', 's', 'root', 'romeos', 'rock', 'repair', 'red', 'prize', 'prayer', 'pleasant', 'plead', 'piteous', 'performd', 'passd', 'paper', 'palace', 'pair', 'opinion', 'obedience', 'nobles', 'neighbour', 'motion', 'miserable', 'lip', 'legs', 'lamb', 'innocent', 'hurt', 'hollow', 'hereafter', 'happiness', 'hail', 'guest', 'gross', 'goodness', 'gage', 'forsworn', 'forgive', 'flight', 'finger', 'fairly', 'faint', 'exile', 'enjoy', 'empty', 'embrace', 'drum', 'dread', 'dishonour', 'deserves', 'denied', 'demand', 'deliverd', 'dares', 'damned', 'custom', 'courtesy', 'contrary', 'continue', 'commanded', 'comest', 'colours', 'christian', 'boatswain', 'blush', 'beyond', 'beloved', 'belly', 'beheld', 'balthasar', 'babe', 'add', 'accuse', 'accept', 'wrongd', 'wounded', 'witherd', 'winter', 'ways', 'waters', 'unnatural', 'unknown', 'titus', 'thunder', 'thrust', 'thourt', 'theirs', 'terror', 'suspicion', 'supper', 'story', 'standing', 'spoken', 'spend', 'shut', 'shortly', 'services', 'senate', 'seeming', 'seas', 'salisbury', 'room', 'richards', 'resign', 'renowned', 'remembrance', 'remains', 'prey', 'praise', 'peril', 'perform', 'painted', 'outward', 'oclock', 'needful', 'natural', 'naked', 'misery', 'low', 'led', 'leads', 'large', 'language', 'lament', 'kingly', 'kin', 'joys', 'joyful', 'joind', 'join', 'horses', 'hopes', 'hid', 'hatred', 'hat', 'haply', 'hap', 'grows', 'government', 'glass', 'gate', 'fourth', 'forbear', 'flies', 'ferdinand', 'feard', 'fame', 'faces', 'excellent', 'endure', 'ends', 'ely', 'durst', 'dukedom', 'drums', 'drown', 'disposition', 'dinner', 'dies', 'delay', 'deal', 'curst', 'curses', 'crownd', 'credit', 'condemnd', 'cleomenes', 'cease', 'camest', 'called', 'caliban', 'bodies', 'big', 'beast', 'bearing', 'bagot', 'assurance', 'apparel', 'advantage', 'access', 'youre', 'worn', 'winters', 'winds', 'willoughby', 'wept', 'wed', 'wants', 'volsce', 'usurp', 'urge', 'unhappy', 'tybalts', 'twill', 'trick', 'tremble', 'trade', 'tomb', 'thief', 'terms', 'takes', 'table', 'surrey', 'stain', 'spite', 'smiles', 'slept', 'sirs', 'sink', 'sigh', 'sicilia', 'shepherds', 'shape', 'serves', 'sends', 'selfsame', 'scope', 'scene', 'sayst', 'rule', 'ruin', 'rogue', 'ring', 'reverend', 'revenged', 'respected', 'resolution', 'realm', 'raise', 'quench', 'puts', 'protest', 'proper', 'powers', 'plot', 'playd', 'plant', 'perpetual', 'perfect', 'passion', 'passage', 'odds', 'number', 'nought', 'naples', 'monstrous', 'mockd', 'minister', 'mild', 'mend', 'manners', 'lusty', 'looking', 'knife', 'knaves', 'kindly', 'isle', 'intent', 'instruments', 'instrument', 'instruct', 'houses', 'hateful', 'hardly', 'granted', 'glorious', 'giving', 'garden', 'friendly', 'french', 'freely', 'frame', 'forced', 'folly', 'finds', 'fail', 'east', 'drops', 'dorcas', 'devils', 'desires', 'degree', 'debt', 'contract', 'contented', 'conduct', 'committed', 'colour', 'clear', 'charged', 'chair', 'calm', 'burthen', 'brings', 'below', 'begun', 'armour', 'approach', 'alike', 'affections', 'accursed', 'absent', 'able', 'yond', 'wrought', 'worlds', 'wondrous', 'willingly', 'whip', 'warlike', 'vows', 'virtues', 'villains', 'veins', 'vault', 'urged', 'uncles', 'tush', 'treacherous', 'tide', 'thrice', 'thinking', 'theyll', 'tells', 'taught', 'tarry', 'tame', 'supposed', 'study', 'streets', 'stoop', 'stabbd', 'special', 'soundly', 'sooth', 'slow', 'slay', 'slaves', 'silent', 'shrew', 'servants', 'saying', 'sanctuary', 'sail', 'round', 'raised', 'purse', 'pure', 'prosperous', 'pronounce', 'princess', 'plainly', 'pieces', 'perish', 'perforce', 'peoples', 'peers', 'patricians', 'partly', 'offended', 'offend', 'nobly', 'month', 'mile', 'methought', 'marketplace', 'loyal', 'lovely', 'loud', 'lion', 'lark', 'jove', 'jack', 'ireland', 'imprisonment', 'imagine', 'huntsman', 'holds', 'heed', 'happily', 'hanged', 'groans', 'grievous', 'grieve', 'greatest', 'graves', 'graces', 'gift', 'gently', 'gaoler', 'forthwith', 'fore', 'firm', 'feeling', 'feed', 'fears', 'fairer', 'express', 'estate', 'equal', 'enmity', 'enforce', 'elder', 'dying', 'dogs', 'dew', 'desert', 'daily', 'dagger', 'cousins', 'contempt', 'consider', 'conceit', 'commission', 'bury', 'boys', 'bow', 'boots', 'boot', 'bodys', 'boar', 'bethink', 'behalf', 'bastard', 'bade', 'avoid', 'assured', 'arrived', 'armd', 'apparent', 'apollo', 'amongst', 'alonso', 'afternoon', 'afford', 'aedile', 'advice', 'adrian', 'accused', 'according', 'aboard', 'younger', 'yoke', 'yielded', 'wreck', 'willt', 'wifes', 'wholesome', 'weeps', 'weather', 'waste', 'waking', 'vienna', 'vaughan', 'tyranny', 'twain', 'treasure', 'talkd', 'surely', 'suppose', 'supply', 'sue', 'stuff', 'stroke', 'strive', 'stoppd', 'steeds', 'staind', 'spur', 'spit', 'sovereignty', 'sore', 'slaughter', 'sky', 'sire', 'single', 'sickness', 'sharp', 'sets', 'served', 'sell', 'seldom', 'sees', 'scape', 'satisfy', 'salt', 'ruled', 'rosaline', 'romans', 'rights', 'reward', 'reverence', 'retire', 'remove', 'regard', 'regal', 'refuse', 'rash', 'rascal', 'purchase', 'protector', 'profit', 'prevent', 'powerful', 'pound', 'pomfret', 'points', 'pluckd', 'pierce', 'persuade', 'pawn', 'parting', 'parliament', 'pack', 'odd', 'obedient', 'oaths', 'north', 'newly', 'neighbours', 'native', 'muster', 'mouths', 'mourn', 'morn', 'milan', 'merit', 'melancholy', 'meeting', 'meant', 'loath', 'lift', 'lieutenant', 'lesser', 'lent', 'jealous', 'island', 'iron', 'inform', 'increase', 'impossible', 'image', 'honourd', 'hic', 'heirs', 'hasty', 'griefs', 'greet', 'grandam', 'gloucesters', 'fruit', 'froward', 'fright', 'forswear', 'forfeit', 'food', 'followd', 'fitzwater', 'fires', 'feet', 'fancy', 'famous', 'falsehood', 'faithful', 'exton', 'exchange', 'entertainment', 'dwell', 'disgrace', 'discover', 'direct', 'dignity', 'devotion', 'determined', 'defence', 'current', 'cup', 'counterfeit', 'constant', 'conspirator', 'conference', 'complain', 'commons', 'chase', 'caught', 'carlisle', 'brook', 'bred', 'breathed', 'brain', 'bore', 'books', 'bird', 'betimes', 'berkeley', 'beholding', 'began', 'beasts', 'beard', 'barren', 'bare', 'babes', 'aunt', 'attempt', 'assure', 'assist', 'arise', 'appointed', 'appetite', 'ant', 'anointed', 'allow', 'alliance', 'ago', 'abide', 'youngest', 'yould', 'yon', 'yields', 'writing', 'wooing', 'wishd', 'wipe', 'windows', 'willing', 'william', 'widows', 'wide', 'whereto', 'westmoreland', 'weight', 'weakness', 'washd', 'wanting', 'wakes', 'wake', 'wait', 'wager', 'vouchsafe', 'visage', 'vessel', 'verily', 'venom', 'utter', 'tullus', 'trusty', 'trumpet', 'trembling', 'torment', 'top', 'tooth', 'toad', 'thread', 'thirty', 'thieves', 'thereby', 'tempest', 'task', 'tapster', 'tall', 'sweeter', 'sway', 'suspect', 'summers', 'sullen', 'suits', 'suitors', 'suitor', 'subtle', 'stronger', 'strokes', 'stranger', 'store', 'stephen', 'stays', 'star', 'spy', 'speakst', 'sovereigns', 'somewhat', 'smooth', 'slip', 'sleeping', 'sights', 'shadows', 'seemd', 'scroop', 'sceptre', 'saints', 'rose', 'ripe', 'redress', 'rebellion', 'reap', 'ran', 'public', 'proclaimd', 'privilege', 'possessd', 'policy', 'pleasing', 'places', 'pitiful', 'penitent', 'pause', 'particular', 'opposite', 'ope', 'oak', 'notice', 'necessity', 'natures', 'mystery', 'mount', 'monument', 'modesty', 'mischance', 'minds', 'memory', 'marks', 'markd', 'mar', 'manage', 'maintain', 'lovest', 'lovers', 'limit', 'likely', 'length', 'league', 'knowing', 'journey', 'jot', 'jewel', 'italy', 'instruction', 'infection', 'hunt', 'human', 'household', 'holp', 'herald', 'heavenly', 'heat', 'heartily', 'hated', 'harsh', 'hangs', 'growing', 'groom', 'groan', 'greatness', 'gaze', 'garland', 'frowns', 'fourteen', 'foreign', 'feather', 'fairest', 'extremity', 'expect', 'exercise', 'executioner', 'executed', 'execute', 'entertain', 'enterd', 'emilia', 'eleven', 'edge', 'earnest', 'dowry', 'dismal', 'dishonourd', 'dish', 'direction', 'determine', 'design', 'descent', 'delight', 'deceit', 'dearly', 'dearest', 'dam', 'craves', 'countrymen', 'conceive', 'clothes', 'civil', 'choler', 'cheerly', 'catch', 'burst', 'bridegroom', 'boldly', 'birds', 'bigger', 'beggars', 'beauteous', 'bastards', 'ass', 'appeal', 'amazed', 'aged', 'adversaries', 'adventure', 'advance', 'ado', 'acquaint', 'abused', 'absolute', 'zeal', 'written', 'wore', 'wolf', 'wisely', 'wills', 'whereon', 'whateer', 'weigh', 'wears', 'weapons', 'weapon', 'wealth', 'waves', 'warriors', 'violence', 'velvet', 'unworthy', 'unjust', 'unfold', 'turnd', 'troops', 'triumphant', 'trespass', 'torture', 'thumb', 'threatening', 'tempt', 'temper', 'telling', 'taking', 'takest', 'sweetly', 'sweat', 'swain', 'sup', 'suddenly', 'succession', 'substance', 'strew', 'sting', 'stern', 'steps', 'stayd', 'statue', 'stale', 'spread', 'spleen', 'south', 'sought', 'sooner', 'solemn', 'snow', 'sitting', 'sings', 'shrift', 'shrewd', 'shook', 'shines', 'shield', 'shell', 'share', 'shapes', 'shallow', 'sepulchre', 'senses', 'secure', 'season', 'sadness', 'ruthless', 'rue', 'ride', 'revolt', 'rescue', 'reputation', 'repose', 'removed', 'remorse', 'rejoice', 'redeem', 'record', 'rebels', 'rare', 'rain', 'rail', 'purge', 'provoked', 'provided', 'provide', 'prosper', 'proportion', 'profess', 'proclamation', 'prime', 'price', 'prevented', 'prevaild', 'prettiest', 'praises', 'possession', 'possess', 'pitch', 'piercing', 'persons', 'pernicious', 'parted', 'paid', 'ourself', 'oh', 'offices', 'officers', 'offerd', 'object', 'nobility', 'nest', 'musicians', 'murderd', 'moves', 'months', 'montagues', 'modest', 'mistrust', 'miss', 'mirth', 'minute', 'mildly', 'midnight', 'metal', 'merely', 'mere', 'meddle', 'mars', 'madness', 'lower', 'lover', 'looked', 'livery', 'limbs', 'liking', 'likeness', 'lightning', 'licio', 'lean', 'laws', 'lately', 'lad', 'knot', 'joints', 'intelligence', 'inquire', 'injury', 'infant', 'inclined', 'immortal', 'ignorance', 'hungry', 'hung', 'holding', 'hire', 'hers', 'herefords', 'henceforth', 'heels', 'hare', 'habit', 'guilt', 'guests', 'grew', 'grain', 'graced', 'goods', 'gladly', 'giddy', 'gardener', 'gallant', 'forty', 'fish', 'fierce', 'fiend', 'fearing', 'favours', 'fashion', 'fardel', 'falls', 'escape', 'enforced', 'endured', 'dozen', 'dispersed', 'disease', 'disdain', 'discourse', 'dion', 'destruction', 'despised', 'depose', 'courteous', 'count', 'couldst', 'corrupt', 'convey', 'conjure', 'confound', 'condemned', 'compare', 'commonwealth', 'clamour', 'churchyard', 'chief', 'chaste', 'charm', 'character', 'challenge', 'censure', 'carried', 'careful', 'caps', 'cap', 'cambio', 'caitiff', 'caesar', 'butcher', 'burnt', 'burns', 'burning', 'brown', 'breed', 'bread', 'brace', 'bottom', 'bosoms', 'block', 'bliss', 'blessing', 'bitterly', 'bidding', 'beshrew', 'behavior', 'battles', 'bands', 'balm', 'ballad', 'backs', 'axe', 'archidamus', 'archbishop', 'apothecary', 'answerd', 'angel', 'amiss', 'ambitious', 'agree', 'ages', 'affords', 'advised', 'actions', 'accident', 'abuse', 'zounds', 'youthful', 'worldly', 'witch', 'wiltshire', 'whit', 'whisper', 'west', 'week', 'wednesday', 'weddingday', 'watchd', 'ward', 'wandering', 'waked', 'venice', 'varlet', 'vanity', 'usurping', 'usurpd', 'upright', 'upont', 'unkindness', 'tyrannous', 'twould', 'tutor', 'trunk', 'troubled', 'troop', 'tricks', 'trees', 'treasons', 'towns', 'touching', 'torch', 'toil', 'tired', 'tie', 'thin', 'term', 'tempted', 'temples', 'swore', 'swears', 'sufferd', 'substitute', 'stout', 'storms', 'stops', 'stolen', 'stock', 'step', 'stamp', 'springs', 'spoil', 'speaking', 'sounds', 'sounded', 'sole', 'sold', 'smell', 'sith', 'silly', 'signify', 'signal', 'sides', 'shroud', 'shoulder', 'shot', 'shop', 'shoes', 'ships', 'settled', 'sentenced', 'senseless', 'seized', 'seeking', 'seated', 'seald', 'scatterd', 'scandal', 'saved', 'satisfaction', 'safely', 'safeguard', 'runs', 'royalty', 'rouse', 'robes', 'riches', 'revenges', 'revel', 'retired', 'reply', 'reconcile', 'recompense', 'readiness', 'reach', 'ravenspurgh', 'rate', 'putting', 'push', 'pursue', 'purple', 'punish', 'proved', 'prosperity', 'profane', 'procure', 'process', 'prick', 'prevail', 'presume', 'pressd', 'preserve', 'poverty', 'pin', 'pilgrimage', 'philosophy', 'petty', 'petition', 'perchance', 'peevish', 'paul', 'partner', 'parties', 'parents', 'owes', 'outrage', 'ought', 'omit', 'occupation', 'nuptial', 'noted', 'nobleness', 'nathaniel', 'mutinous', 'murders', 'mowbrays', 'minola', 'ministers', 'mildness', 'mightst', 'middle', 'members', 'meantime', 'mates', 'matchd', 'marvel', 'lying', 'lovel', 'longs', 'load', 'livest', 'lists', 'likewise', 'likes', 'liberal', 'learnd', 'lap', 'landed', 'lamentable', 'lambs', 'knowt', 'kisses', 'kinsmen', 'kingdoms', 'keys', 'joves', 'jesu', 'intents', 'intended', 'integrity', 'instantly', 'injurious', 'infected', 'infect', 'ice', 'hostess', 'hopeful', 'homely', 'herein', 'herd', 'helps', 'heavier', 'hazard', 'harp', 'hangd', 'grieves', 'grieved', 'greeting', 'grandsire', 'goose', 'godden', 'gifts', 'ghostly', 'fully', 'frozen', 'fray', 'framed', 'frail', 'fondly', 'flood', 'flattering', 'fingers', 'filld', 'figure', 'felt', 'falsely', 'extreme', 'except', 'example', 'estimation', 'esteem', 'errand', 'entreaties', 'entrance', 'ended', 'employd', 'eldest', 'eaten', 'earthly', 'drunken', 'drinking', 'dreamd', 'draws', 'downright', 'dower', 'dispatchd', 'dismiss', 'discontented', 'discharge', 'dire', 'dine', 'dido', 'devilish', 'destroy', 'designs', 'descend', 'departure', 'defy', 'deck', 'dearer', 'deaf', 'darkness', 'damnd', 'crying', 'cowardly', 'coventry', 'couple', 'counsels', 'cost', 'correction', 'coronation', 'conveyd', 'contents', 'conquest', 'conclude', 'complexion', 'complaint', 'compass', 'companion', 'commit', 'commands', 'coat', 'cloudy', 'circumstance', 'chosen', 'childrens', 'cherish', 'chat', 'charges', 'capulets', 'busy', 'burgundy', 'breathing', 'brains', 'bowels', 'bowd', 'bootless', 'bond', 'boldness', 'blown', 'bleeding', 'betide', 'beside', 'bereft', 'bent', 'beget', 'beef', 'beautys', 'beats', 'beaten', 'beams', 'baptistas', 'aspect', 'ashamed', 'array', 'apt', 'approbation', 'apace', 'amorous', 'amends', 'amain', 'also', 'alls', 'allegiance', 'aim', 'afraid', 'acquainted', 'accusation', 'abraham', 'wronged', 'wrinkled', 'wring', 'worships', 'worser', 'worms', 'worm', 'wonderful', 'wolves', 'wither', 'wink', 'window', 'whoreson', 'whispering', 'whipt', 'whatsoever', 'wet', 'wenches', 'wedding', 'wedded', 'watery', 'wander', 'waits', 'vowd', 'volscians', 'volscian', 'visitation', 'virgin', 'vices', 'vex', 'venture', 'vent', 'vast', 'utmost', 'unseen', 'unruly', 'undo', 'tunis', 'trow', 'trifles', 'tree', 'train', 'torn', 'tops', 'toads', 'tied', 'thrown', 'throng', 'threw', 'thousands', 'thereto', 'theme', 'thankful', 'tewksbury', 'tenderness', 'temple', 'taunts', 'tarpeian', 'tardy', 'talking', 'sycorax', 'swelling', 'sweetest', 'swallow', 'sunday', 'sufficient', 'suffering', 'sufferance', 'stumble', 'stuffd', 'strife', 'strict', 'stomach', 'stoln', 'stocks', 'stiff', 'stick', 'steed', 'stead', 'statutes', 'spurs', 'spiders', 'speedy', 'song', 'softly', 'smotherd', 'smallest', 'slily', 'sleeps', 'slanderous', 'skill', 'sixteen', 'silk', 'shun', 'shown', 'showing', 'shoulders', 'shores', 'sheepshearing', 'sheep', 'shade', 'setting', 'serious', 'sensible', 'seet', 'seeks', 'secrets', 'search', 'scratch', 'scotland', 'scorns', 'score', 'schoolmaster', 'savage', 'sadly', 'saddle', 'sacrament', 'rust', 'rushes', 'royalties', 'roundly', 'roses', 'roof', 'rode', 'rob', 'roaring', 'roard', 'roar', 'revenue', 'revengeful', 'returnd', 'rests', 'resist', 'requires', 'reported', 'repeald', 'reigns', 'redemption', 'recreant', 'ransom', 'race', 'pursuivant', 'pursuit', 'pursued', 'proudest', 'prophet', 'prophesy', 'prophecy', 'proceedings', 'privy', 'prisoners', 'press', 'prating', 'prate', 'popular', 'pomp', 'plebeians', 'plays', 'plants', 'physic', 'perjury', 'perjured', 'perfection', 'pembroke', 'pedlar', 'pearl', 'pawnd', 'pattern', 'patrician', 'pace', 'otherwise', 'ornaments', 'orator', 'oppression', 'ons', 'offences', 'numbers', 'northern', 'nod', 'nobleman', 'nimble', 'nightly', 'nigh', 'neptune', 'ned', 'named', 'nails', 'mustard', 'murderous', 'murdered', 'multitude', 'moving', 'moody', 'monster', 'moiety', 'moe', 'model', 'minutes', 'midst', 'mewd', 'mercutios', 'merciful', 'merchant', 'meed', 'meaner', 'matters', 'marvellous', 'mariners', 'margarets', 'manifest', 'manhood', 'main', 'maiden', 'madest', 'lute', 'lunatic', 'loyalty', 'lowly', 'lodging', 'lodged', 'lodge', 'loathsome', 'loathed', 'lions', 'line', 'likelihood', 'lights', 'lief', 'lick', 'lewd', 'level', 'lenity', 'leg', 'leaving', 'learning', 'leaden', 'lasting', 'lamentation', 'lads', 'knocks', 'knight', 'kites', 'kissd', 'jupiter', 'joint', 'inward', 'intends', 'insulting', 'instant', 'ins', 'innocence', 'injustice', 'inch', 'incensed', 'imagination', 'ignoble', 'humility', 'honey', 'history', 'hinder', 'hill', 'helm', 'heinous', 'height', 'heigh', 'heel', 'harmony', 'harbour', 'happier', 'hangman', 'guiltless', 'growth', 'grim', 'gloves', 'glasses', 'game', 'gall', 'friendship', 'freedom', 'frantic', 'fouler', 'forsooth', 'forsake', 'forgiveness', 'forehead', 'following', 'folks', 'flourish', 'flock', 'flint', 'flatterers', 'fights', 'fifth', 'fifteen', 'feasts', 'fawn', 'fat', 'falling', 'fallen', 'faced', 'extremes', 'evils', 'evermore', 'event', 'eternal', 'error', 'entreaty', 'easily', 'eagle', 'eager', 'drownd', 'drift', 'dried', 'drew', 'dressd', 'dreamt', 'dram', 'downfall', 'doves', 'doubtful', 'divided', 'dissemble', 'disgraced', 'discontent', 'dined', 'din', 'dim', 'diest', 'destroyd', 'destiny', 'delicate', 'deer', 'dearth', 'date', 'dash', 'dared', 'dancing', 'damnable', 'cur', 'crush', 'crows', 'crow', 'crossd', 'crept', 'creatures', 'craft', 'cracking', 'crackd', 'cover', 'courtier', 'counsellor', 'corruption', 'cords', 'cook', 'consuls', 'constable', 'consorted', 'considered', 'consequence', 'conqueror', 'confirm', 'confidence', 'confessor', 'confession', 'condemn', 'conclusion', 'concerns', 'commends', 'comforts', 'cock', 'coast', 'cloud', 'closed', 'clock', 'cliffords', 'claudios', 'chose', 'childish', 'chiefest', 'cheque', 'cheap', 'changing', 'chances', 'certainly', 'ceremonious', 'causes', 'carthage', 'carries', 'careless', 'calamity', 'cabin', 'butchers', 'burden', 'build', 'brotherhood', 'brittany', 'bridal', 'breasts', 'breaks', 'breaking', 'braved', 'brat', 'branches', 'branch', 'bowl', 'bounty', 'boisterous', 'boast', 'blushing', 'blue', 'blot', 'blaze', 'bending', 'believed', 'begot', 'beggd', 'became', 'beating', 'bay', 'bar', 'backd', 'awaked', 'avoided', 'attended', 'asking', 'ashes', 'armed', 'approved', 'apprehend', 'appears', 'apollos', 'apart', 'antium', 'anothers', 'angels', 'aloud', 'alms', 'ale', 'albans', 'airy', 'aims', 'agreed', 'afore', 'affliction', 'affect', 'afeard', 'advocate', 'advanced', 'admit', 'acts', 'achieve', 'abhorrd', 'yorks', 'yielding', 'yere', 'yare', 'wrangling', 'worthily', 'worshipful', 'wooers', 'wonders', 'wonderd', 'woeful', 'witnesses', 'wishing', 'wisest', 'winged', 'wing', 'whoever', 'whistle', 'western', 'welshmen', 'welcomes', 'weighty', 'weighd', 'weal', 'wayward', 'wave', 'warrants', 'warning', 'walter', 'wales', 'vouch', 'volume', 'vineyard', 'villany', 'victorious', 'vassal', 'value', 'utterance', 'using', 'unlawful', 'unfit', 'uneven', 'understanding', 'underneath', 'uncertain', 'unborn', 'ugly', 'turning', 'turned', 'truer', 'troy', 'troublous', 'triumphs', 'tried', 'tribute', 'tribune', 'trembles', 'treachery', 'transported', 'toy', 'tos', 'torments', 'tonguetied', 'token', 'toe', 'titles', 'tire', 'timeless', 'thwack', 'throats', 'threats', 'threaten', 'thorns', 'thirst', 'thereon', 'testimony', 'territories', 'terrible', 'tents', 'tenderly', 'tend', 'temperance', 'tamed', 'tales', 'taket', 'tainted', 'tail', 'swoon', 'sweets', 'surfeit', 'supreme', 'support', 'sunshine', 'suns', 'sung', 'summon', 'suck', 'succor', 'succeed', 'strumpet', 'strongly', 'stretch', 'street', 'stream', 'straw', 'stooping', 'stirs', 'stirring', 'stirrd', 'stem', 'steepd', 'stark', 'stampd', 'staff', 'spurn', 'spilt', 'spider', 'spices', 'sped', 'solicit', 'solemnity', 'solely', 'society', 'smiling', 'sleepy', 'slanders', 'skulls', 'sixth', 'sisters', 'sisterhood', 'silken', 'signs', 'sighd', 'siege', 'shrink', 'showers', 'shine', 'shamed', 'severe', 'serpent', 'scurvy', 'scornd', 'scolding', 'scold', 'scholar', 'sceptres', 'scars', 'scarlet', 'scarcely', 'sayest', 'saucy', 'sat', 'sap', 'sands', 'sailors', 'safer', 'sack', 'royally', 'rocks', 'rites', 'rising', 'rewards', 'revive', 'returned', 'restraint', 'restored', 'resort', 'resolute', 'requite', 'required', 'require', 'reproof', 'reprieve', 'repossess', 'replied', 'repetition', 'repeal', 'renownd', 'render', 'rend', 'rememberd', 'remedies', 'relish', 'relent', 'regiment', 'recover', 'reckoning', 'receives', 'rebuke', 'rebel', 'rear', 'rats', 'rascals', 'rapiers', 'rancour', 'raiment', 'raging', 'ragged', 'rages', 'rack', 'quake', 'puppet', 'punishd', 'puissant', 'publicly', 'prunes', 'pronounced', 'profits', 'produce', 'print', 'preserved', 'prerogative', 'preposterous', 'precedent', 'prays', 'praised', 'pox', 'pour', 'pounds', 'pot', 'posterity', 'possessed', 'poisonous', 'pocket', 'plucks', 'pledge', 'plate', 'plaints', 'pitied', 'pine', 'pinchd', 'pilot', 'physicians', 'phoebus', 'philip', 'phaethon', 'peters', 'pestilence', 'persuasion', 'permit', 'peremptory', 'pent', 'penitence', 'peaceful', 'pays', 'path', 'pate', 'pastime', 'partake', 'pardons', 'pardond', 'painting', 'orphans', 'opposed', 'obeyd', 'nursed', 'notwithstanding', 'notorious', 'notes', 'noses', 'noon', 'nightingale', 'nice', 'newmade', 'neglect', 'needless', 'necessary', 'neat', 'nearer', 'naught', 'narrow', 'napkin', 'nail', 'mutiny', 'musty', 'murdering', 'mum', 'mouse', 'mourning', 'mounting', 'mountain', 'mould', 'mornings', 'moral', 'mongst', 'monarch', 'mocking', 'mockery', 'moan', 'mistaking', 'misshapen', 'misfortune', 'miseries', 'minded', 'millions', 'milk', 'miles', 'midwife', 'merriment', 'merrily', 'melt', 'medicine', 'meal', 'mast', 'marrying', 'marrd', 'marble', 'madman', 'madly', 'lustful', 'lungs', 'louder', 'longest', 'loins', 'lodowick', 'locks', 'limits', 'lightness', 'liar', 'levy', 'lends', 'lecture', 'leap', 'latest', 'lance', 'ladyship', 'lace', 'knowest', 'knit', 'knightly', 'knighthood', 'kneeld', 'kissing', 'killing', 'killed', 'key', 'kent', 'keepst', 'juliets', 'jocund', 'jealousies', 'jade', 'issued', 'interchange', 'instead', 'insolence', 'ink', 'injuries', 'inferior', 'inconstant', 'inclination', 'incapable', 'impression', 'importune', 'import', 'imperial', 'impediment', 'impatient', 'impatience', 'immediately', 'idly', 'hurried', 'hurl', 'hourly', 'host', 'horn', 'homage', 'hole', 'hoar', 'highway', 'higher', 'hideous', 'hercules', 'herbs', 'heaviness', 'heartblood', 'hearst', 'hears', 'headstrong', 'haunt', 'haughty', 'harvest', 'harmless', 'hardhearted', 'halt', 'hall', 'hag', 'gulf', 'greetings', 'greece', 'grandfather', 'governor', 'govern', 'gost', 'gossips', 'goddess', 'glories', 'getting', 'gentler', 'gape', 'gap', 'gaol', 'gait', 'g', 'function', 'fruitful', 'frost', 'front', 'frighted', 'fourscore', 'fortunate', 'fors', 'forest', 'forces', 'footman', 'follies', 'flow', 'fleet', 'flayed', 'flame', 'fixd', 'fitted', 'fits', 'firmly', 'fills', 'fever', 'festival', 'female', 'feeding', 'fee', 'fate', 'fasting', 'farthest', 'farm', 'famish', 'familiar', 'faintly', 'extremest', 'expedition', 'events', 'especially', 'eret', 'entrails', 'enjoyd', 'employ', 'embraced', 'elsewhere', 'eight', 'eating', 'earths', 'earldom', 'eagles', 'dove', 'doubly', 'doricles', 'doors', 'divorce', 'division', 'divines', 'divide', 'distressed', 'dispose', 'displeasure', 'disguised', 'discords', 'discharged', 'direful', 'digest', 'difference', 'dian', 'diadem', 'device', 'den', 'demands', 'delivered', 'delights', 'degrees', 'deeply', 'deeper', 'deceive', 'decay', 'debase', 'daylight', 'dangers', 'dame', 'curtain', 'curious', 'crew', 'created', 'create', 'coz', 'cowardice', 'covert', 'counted', 'councils', 'cordial', 'controversy', 'consumed', 'conquer', 'confines', 'conditions', 'concluded', 'conceal', 'compound', 'composition', 'compassion', 'commanding', 'comfortable', 'coldly', 'coals', 'closely', 'cloak', 'climate', 'clean', 'circumstances', 'christopher', 'chop', 'choleric', 'cheerfully', 'chastity', 'chastisement', 'changes', 'chamberlain', 'chains', 'centre', 'cave', 'caused', 'cat', 'car', 'captive', 'capital', 'canopy', 'calumny', 'calais', 'butt', 'bush', 'budge', 'broad', 'bribe', 'breeding', 'breathes', 'brawling', 'brawl', 'brass', 'bows', 'bought', 'borrow', 'boon', 'bondage', 'bolingbrokes', 'bodes', 'bloods', 'blessings', 'blessd', 'blemish', 'bleeds', 'blade', 'bishops', 'bill', 'biancas', 'beware', 'betray', 'bet', 'bestride', 'bestow', 'bell', 'beguiled', 'beguile', 'befits', 'befall', 'befal', 'beautiful', 'beastly', 'bate', 'baseness', 'banquet', 'bankrupt', 'bail', 'baggage', 'backward', 'bachelor', 'awry', 'attach', 'assistance', 'assault', 'ashore', 'ascend', 'arrest', 'apprehension', 'apply', 'appeared', 'ape', 'antic', 'antiates', 'amity', 'ambition', 'amazement', 'altogether', 'alterd', 'aloof', 'allies', 'aimd', 'afoot', 'afflict', 'aediles', 'advertised', 'adverse', 'admitted', 'added', 'actor', 'acknowledge', 'achieved', 'abbot', 'yesternight', 'yellow', 'yard', 'wrongfully', 'wretchedness', 'worthiest', 'worthier', 'wondering', 'womens', 'wombs', 'womanish', 'witty', 'witht', 'witchcraft', 'wishes', 'wilful', 'wildly', 'widower', 'whore', 'whipped', 'whereupon', 'whereer', 'whereby', 'wheels', 'wheel', 'whatever', 'westminster', 'weepst', 'weed', 'web', 'wearing', 'wealthy', 'wax', 'watching', 'watchful', 'wasted', 'wary', 'warn', 'warmd', 'wantons', 'waning', 'walks', 'waken', 'wailing', 'vulgar', 'visor', 'visiting', 'violate', 'vigour', 'victors', 'vial', 'vexd', 'vexation', 'verity', 'verge', 'venge', 'veil', 'vail', 'utterd', 'usurps', 'usage', 'urging', 'upper', 'upons', 'unwilling', 'unwieldy', 'unusual', 'untainted', 'unsheathe', 'unrest', 'unprovided', 'unpleasing', 'unluckily', 'unlookd', 'unlikely', 'unlike', 'unjustly', 'unity', 'ungovernd', 'ungentle', 'unfeigned', 'undergo', 'unawares', 'unavoided', 'unaccustomd', 'ulysses', 'tyrants', 'turtle', 'turks', 'tunes', 'trudge', 'trod', 'trimmd', 'trim', 'trenches', 'trencher', 'treble', 'treaty', 'travel', 'transport', 'tragic', 'traffic', 'toys', 'toucheth', 'touches', 'torches', 'tongueless', 'tigers', 'tiger', 'tides', 'throughly', 'threshold', 'threes', 'threepence', 'threefold', 'threat', 'thorny', 'thorn', 'thirsty', 'thinkest', 'thick', 'theyre', 'tending', 'tellus', 'teeming', 'teaching', 'talkst', 'talked', 'swim', 'swiftly', 'swifter', 'swell', 'swearing', 'swallowd', 'susan', 'surprised', 'surname', 'surly', 'suppliant', 'supple', 'suppertime', 'sunder', 'sum', 'suggest', 'sugar', 'suffice', 'sues', 'sued', 'suckd', 'successful', 'suburbs', 'subscribe', 'submission', 'subdued', 'subdue', 'stumbled', 'studies', 'studied', 'stuck', 'stubborn', 'stripes', 'stride', 'stratagems', 'strangely', 'stoopd', 'stole', 'stinking', 'steward', 'steterat', 'stealing', 'starve', 'starts', 'standst', 'stage', 'stabbed', 'sprites', 'spotted', 'spoils', 'split', 'spies', 'spell', 'speedily', 'speechless', 'speeches', 'spectacle', 'spear', 'sparkling', 'space', 'soonest', 'soninlaw', 'songs', 'somebody', 'solace', 'soil', 'soar', 'smoking', 'smiled', 'smelt', 'slippd', 'slight', 'slide', 'slender', 'slaughterhouse', 'slack', 'skins', 'singular', 'simois', 'signories', 'sigeia', 'sickly', 'shuts', 'shunnd', 'shriek', 'shower', 'shouts', 'shout', 'shorten', 'shock', 'shift', 'shelter', 'sheets', 'shamest', 'shameless', 'sex', 'severity', 'setst', 'session']
```

## Embedding과 vocab 정보를 tsv 파일로 만들고 이를 저장합니다.

```python
out_v = io.open('vectors.tsv', 'w', encoding='utf-8')
out_m = io.open('metadata.tsv', 'w', encoding='utf-8')

for index, word in enumerate(vocab):
  if index == 0:
    continue  # skip 0, it's padding.
  vec = weights[index]
  out_v.write('\t'.join([str(x) for x in vec]) + "\n")
  out_m.write(word + "\n")
out_v.close()
out_m.close()
```

```python
try:
  from google.colab import files
  files.download('vectors.tsv')
  files.download('metadata.tsv')
except Exception:
  pass
```
