# K_Contemp_Poem_Generator

## ๐จโ๐ ์ค๋์ฃผ์ ๋ฐฑ์์ ๊ฐ์ฑ์ผ๋ก ์ ์ง ๊ทธ๋ด๋ฏํ ํ๋์ ๋ง๋ค๊ธฐ 

Using: <br>

- Google Colab
- Konlpy
- Tensorflow
- LSTM, Bidirectional, Adam, Softmax

์ฌ์ฉํ ๋ฐ์ดํฐ: 

__๋ฐฑ์__: ๋ฐ๋ค, ๋ด๊ฐ ์๊ฐํ๋ ๊ฒ์, ๋ด๊ฐ ์ด๋ ๊ฒ ์ธ๋ฉดํ๊ณ , ๋จ์ ์์ฃผ ์ ๋ ๋ฐ์๋ด๋ฐฉ, ํฐ ๋ฐ๋๋ฒฝ์ด ์์ด, ์ฌ์น, ๋์ ๋ํ์ค์ ํฐ ๋น๋๊ท, ํต์2, ์ฌ์ฐ๋๊ณจ์กฑ, ๊ตญ์

__์ค๋์ฃผ__: ์ํ์, ์ฐธํ๋ก, ํธ์ง, ์์, ์ญ์๊ฐ, ํธ์ฃผ๋จธ๋, ์๋ก์ด ๊ธธ, ๊ฒจ์ธ, ๋ด, ๋, ๋ฌด์์ด ์๊ฐ, ๋ณ ํค๋ ๋ฐค, ์ฌ๋์ค๋ฐ ์ถ์ต, ์ฐํ์ ์คํ, ์๋, ์์ฐ์ ์ธ์ํ, ์กฐ๊ฐ๊ป์ง, ๋ ๋ค๋ฅธ ๊ณ ํฅ, ๊ธธ, ๋ณ์, ์ด ํ๋ 

### 1) Google Colab ํ๊ฒฝ์์ Konlpy ์ธํํ๊ธฐ

```python
!apt-get update
!apt-get install g++ openjdk-8-jdk 
!pip install konlpy JPype1-py3
!bash <(curl -s https://raw.githubusercontent.com/konlpy/konlpy/master/scripts/mecab.sh)
```

๋ก ๋จผ์  ๊ตฌ๊ธ ์ฝ๋ฉ ํ๊ฒฝ์์ konlpy๋ฅผ ์ฌ์ฉํ  ์ ์๊ฒ ์ค์นํด์ค๋๋ค.



### 2) ๋ฐ์ดํฐ๋ฅผ ์ฝ์ด์ Tokenization ์งํํ๊ธฐ

```python
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer()

data = open('./drive/MyDrive/text/k_poem.txt').read()

corpus = data.split("\n")

tokenizer.fit_on_texts(corpus)
total_words = len(tokenizer.word_index) + 1
```

Tokenization์ ํ์ํ๋ก์ฐ์ ์ผ๋ผ์ค๋ก ์งํํฉ๋๋ค. ๊ตฌ๊ธ ๋๋ผ์ด๋ธ๋ฅผ ๋ง์ดํธํ์ฌ ์์งํ ์ ๋ฐ์ดํฐ์์ ๋ถ๋ฌ์ค๊ณ  ์ฝ์ด๋๋๋ค. ์ค ๋จ์๋ก ์ฝ๊ณ  corpus์ ๋ํด tokenization์ ์งํํฉ๋๋ค. total_words์ 1์ ๋ํ๋ ์ด์ ๋ Out-Of-Vocabluary(OOV) token์ ์ด์ฉํ๊ธฐ ๋๋ฌธ์ ์ด๋ฅผ ๋ณด์ถฉํ๊ธฐ ์ํด์ ๋ํฉ๋๋ค.



### 3) n-gram ์ํ์ค๋ฅผ ์์ฑํ๊ณ  ๋ฌธ์ฅ ํจ๋ฉํ๊ธฐ

```python
input_sequences = []
#n-gram ๋จ์๋ก ํ์ต ํ  ์์ .
for line in corpus:
	token_list = tokenizer.texts_to_sequences([line])[0]
	for i in range(1, len(token_list)):
		n_gram_sequence = token_list[:i+1]
		input_sequences.append(n_gram_sequence)

# ๊ฐ์ฅ ๊ธด ์ค์ ๋ฐ๋ผ ๋๋จธ์ง ์ค๋ค๋ ๋ง์ถฐ์ ๊ธธ์ด๋ฅผ ํจ๋ฉํจ
max_sequence_len = max([len(x) for x in input_sequences])
input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))

# create predictors and label
# ์์ธก์ ์ฌ์ฉ๋๋ ์ํ์ค์ ๋ผ๋ฒจ์ ์ค์ . ๋งจ ๋ง์ง๋ง ๋จ์ด.
xs, labels = input_sequences[:,:-1],input_sequences[:,-1]
ys = tf.keras.utils.to_categorical(labels, num_classes=total_words)
```

์ฝ์ด๋ค์ธ ๋ฐ์ดํฐ๋ฅผ ํ์ต ๋ฐ์ดํฐ๋ก ๋ง๋ค๊ธฐ ์ํ ๊ณผ์ ์๋๋ค. ํน์  ๋จ์ด๊ฐ ๋ฑ์ฅํ๋ฉด ์ด๋ ํ ๋จ์ด๊ฐ ๋ค์์ ๋์ฌ ๊ฒ์ธ์ง ์์ธกํ  ์ ์๋๋ก n-gram์ ํํ๋ก ์ํ์ค๋ค์ ๋ง๋ญ๋๋ค. ๊ทธ๋ฆฌ๊ณ  ๊ฐ์ฅ ๊ธด ๋ฌธ์ฅ์ ๊ธฐ์ค์ผ๋ก ๋๋จธ์ง ์ํ์ค๋ค์ ์์ 0์ผ๋ก ํจ๋ฉํฉ๋๋ค.

๋ํ ์ด๋ ๊ฒ ๋ง๋ค์ด์ง ์ํ์ค๋ค์ ์ชผ๊ฐ์ด ์์ธก์ ์ฌ์ฉ๋  ์ํ์ค์ ๋ผ๋ฒจ์ ์ค์ ํฉ๋๋ค. 



### 4) Training Model์ ๋ง๋ค์ด ํ๋ จ์ํค๊ธฐ

```python
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import numpy as np

model = Sequential()
model.add(Embedding(total_words, 100, input_length=max_sequence_len-1))
# 150๊ฐ์ unit์ ๊ฐ๊ณ  ์๋ LSTM layer๋ฅผ ์ถ๊ฐ
model.add(Bidirectional(LSTM(150)))
model.add(Dense(total_words, activation='softmax'))
adam = Adam(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
history = model.fit(xs, ys, epochs=100, verbose=1)
```

์๋ฐฉํฅ LSTM ๋ชจ๋ธ์ ์ฌ์ฉํ์ฌ ๋คํธ์ํฌ๋ฅผ ๊ตฌ์ฑํฉ๋๋ค. Optimizer๋ก๋ Adam, ํ์ฑํจ์๋ softmax, ์์คํจ์๋ก๋ categorical cross entropy๋ฅผ ์ ์ฉํฉ๋๋ค. epoch์๋ 100, learning rate๋ 0.01๋ก ์ค์ ํฉ๋๋ค.

๋ชจ๋ธ์ ์์์ ์ ์ํ xs์ ys๋ฅผ ํผํํ์ฌ training์ ์งํํฉ๋๋ค.



![img](https://drive.google.com/uc?id=1m9Mbc4OUORwL8fkqsx__SotP0g9WmvdH)

๋ชจ๋ธ์ fitํ์ฌ ๋์จ ๊ฒฐ๊ณผ์๋๋ค. ์ต์ข์ ์ผ๋ก 0.93์ ๋์ accuracy๋ฅผ ๋ณด์ฌ์ค๋๋ค.

### 5) ํ๋ จ๋ ๋ชจ๋ธ์ ์ ์ฉํ์ฌ seed text์ ๋ํ ๋ค์ ๋จ์ด๋ค ์์ธกํ๊ธฐ

```python
seed_text = "๋ ์ด ๋ฐ์ต๋๋ค"
next_words = 50
  
for _ in range(next_words):
	token_list = tokenizer.texts_to_sequences([seed_text])[0]
	token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
	predicted = model.predict(token_list, verbose=0).argmax(axis=-1)
	output_word = ""
	for word, index in tokenizer.word_index.items():
		if index == predicted:
			output_word = word
			break
	seed_text += " " + output_word


print(seed_text)
```

```
๋ ์ด ๋ฐ์ต๋๋ค ์์๋ค ์ง๋๋ค ๋ฐค์ ์ด์ ์ถ์ถ์ด ์ค๋ฉด ์์ ์์ ์๋ฐฐ ์์๋ ์์ฌ๋ฐ์ ์๋ค ์์๋ ์๋ผ์ฌ๋ฐ์ ๊ทธ๋ํ ์ฌ๋ฆฌ์์ค๋ ๊ฒ์ด๋ค ๋ชฉ์๋ค ๊ฐ๊น์ด ์ถ๋ค๊ณ  ์๊ฐ ๊ฑฐ์ธ๋ค ์ธ์ ๋ ํ ๋ฐฉ์ ๋ค์ด์ ์ฅ์ ๋ถ์ด์๋ค ๋ค๋ฐ์ ๋์ ๋ณธ๋ค ์ฃผ์ ์์ด์ ๋๋ ๊ฐ๋ํ ์๋ฒ์ง๋ฅผ ๊ฐ์ง ๊ฒ๊ณผ ๊ฒ์ ๋ฌด์์ธ๊ฐ ํฐ ํ๋์ด ์ท์ ์ฌํ์ด๋ฉฐ ํ๋ฉฐ ์ด๋ ๋น๋๊ท ํ๊ณ  ๊ฒ๊ฐ์ด ์ด ์ด
```

seed text๋ก "๋ ์ด ๋ฐ์ต๋๋ค"๋ฅผ ์๋ ฅํ๊ณ  ํ์ 50๊ฐ์ ๋จ์ด๋ฅผ ์ถ๋ ฅํ๊ฒ ํ์์ต๋๋ค. ๋ง์ด ์๋๋ ๋ฏ ํ๋ฉด์๋ ์ฝํ์ง๊ธฐ๋ ํ๋ ์๊ฐ ์์ฑ๋์์ต๋๋ค.

### 6) Konlpy๋ฅผ ํ์ฉํ์ฌ ์์ ๋ฑ์ฅํ๋ ๋ช์ฌ๋ฅผ ์ถ์ถํ์ฌ ์ ๋ชฉ ์ ํ๊ธฐ / ์์ ํํ๋ก ์ถ๋ ฅํ๊ธฐ

```python
import konlpy
from konlpy.tag import Mecab

tokenizer2 = Mecab()
line = []
nouns = []

line = tokenizer2.pos(seed_text)

for word, tag in line:
    if tag in ['NNG']:
        nouns.append(word)
print(nouns)
```

์์ ์ธํํด๋์ konlpy๋ฅผ ํ์ฉํด๋ณด๋ ค ํฉ๋๋ค. Mecab์ ์ฌ์ฉํ์ฌ ๋ง๋ค์ด์ง ์์ ๋ํด ํ์ฌ๋ฅผ ํ๊นํ๊ณ  ๋ช์ฌ๋ฅผ ์ถ์ถํด๋ด๋๋ค.

```
['๋ ', '๋ฐค', '์', '์๋ฐฐ', '์', '์', '์ฌ๋ฐ', '์๋ค', '์', '์๋ผ', '์ฌ๋ฐ', '์ฌ๋ฆฌ', '์', '๋ชฉ์', '์๊ฐ', '์ฅ', '๋ถ์ด', '๋ค', '๋ฐ', '๊ฐ๋', '์๋ฒ์ง', 'ํ๋', '์ท', '์ฌํ', '๋น๋๊ท']
```

```python
import textwrap
import random

title = random.choice(nouns)
result = textwrap.wrap(seed_text, width=30)

print(title)
print('\n')
print('\n'.join(result))
```

์ต์ข์ ์ผ๋ก ์์ ํํ๋ก ์ํ์ ์ถ๋ ฅํ๊ธฐ ์ํด ํ์ด์ฌ์ ๋ผ์ด๋ธ๋ฌ๋ฆฌ๋ค์ ํ์ฉํด๋ด๋๋ค. ํ ์ค์ 30๊ธ์(๋จ์ด ๋จ์๋ก ๋๊น)๋ฅผ ๊ธฐ์ค์ผ๋ก ๋ค์ ์ค๋ก ๋์ด๊ฐ๊ฒ ํ๊ณ , ์์ ์ถ์ถํ ๋ช์ฌ๋ค ์ค ํ๋๋ฅผ ๋๋ค์ผ๋ก ์ ๋ชฉ์ผ๋ก ์ ํ์ฌ ๋ณด์ฌ์ง๊ฒ ํฉ๋๋ค. 

```
์ฌํ


๋ ์ด ๋ฐ์ต๋๋ค ์์๋ค ์ง๋๋ค ๋ฐค์ ์ด์ ์ถ์ถ์ด ์ค๋ฉด
์์ ์์ ์๋ฐฐ ์์๋ ์์ฌ๋ฐ์ ์๋ค ์์๋ ์๋ผ์ฌ๋ฐ์
๊ทธ๋ํ ์ฌ๋ฆฌ์์ค๋ ๊ฒ์ด๋ค ๋ชฉ์๋ค ๊ฐ๊น์ด ์ถ๋ค๊ณ  ์๊ฐ
๊ฑฐ์ธ๋ค ์ธ์ ๋ ํ ๋ฐฉ์ ๋ค์ด์ ์ฅ์ ๋ถ์ด์๋ค ๋ค๋ฐ์
๋์ ๋ณธ๋ค ์ฃผ์ ์์ด์ ๋๋ ๊ฐ๋ํ ์๋ฒ์ง๋ฅผ ๊ฐ์ง ๊ฒ๊ณผ
๊ฒ์ ๋ฌด์์ธ๊ฐ ํฐ ํ๋์ด ์ท์ ์ฌํ์ด๋ฉฐ ํ๋ฉฐ ์ด๋
๋น๋๊ท ํ๊ณ  ๊ฒ๊ฐ์ด ์ด ์ด
```



์์ํ ์กฐ์์ด์ง๋ง ๋ณด๊ธฐ์๋ ์ ์ง ๊ทธ๋ด๋ฏํ ์๊ฐ ์์ฑ๋์์ต๋๋ค. (์)



## ์ฐธ๊ณ 

- [TensorFlow - Natural Language Processing (NLP) Zero to Hero ไธญ](https://www.youtube.com/watch?v=fNxaJsNG3-s&list=PLQY2H8rRoyvzDbLUZkbudP-MFQZwNmU4S)

  - [Training an AI to create poetry (NLP Zero to Hero - Part 6) - YouTube](https://www.youtube.com/watch?v=ZMudJXhsUpY&list=PLQY2H8rRoyvzDbLUZkbudP-MFQZwNmU4S&index=6)

- [07) ์ผ๋ผ์ค(Keras) ํ์ด๋ณด๊ธฐ - ๋ฅ ๋ฌ๋์ ์ด์ฉํ ์์ฐ์ด ์ฒ๋ฆฌ ์๋ฌธ (wikidocs.net)](https://wikidocs.net/32105)
