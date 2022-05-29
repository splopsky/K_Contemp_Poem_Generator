# K_Contemp_Poem_Generator

## 👨‍🎓 윤동주와 백석의 감성으로 왠지 그럴듯한 현대시 만들기 

Using: <br>

- Google Colab
- Konlpy
- Tensorflow
- LSTM, Bidirectional, Adam, Softmax

사용한 데이터: 

__백석__: 바다, 내가 생각하는 것은, 내가 이렇게 외면하고, 남신의주 유동 박시봉방, 흰 바람벽이 있어, 여승, 나와 나타샤와 흰 당나귀, 통영2, 여우난골족, 국수

__윤동주__: 자화상, 참회록, 편지, 서시, 십자가, 호주머니, 새로운 길, 겨울, 봄, 눈, 무서운 시간, 별 헤는 밤, 사랑스런 추억, 산협의 오후, 소년, 아우의 인상화, 조개껍질, 또 다른 고향, 길, 병원, 초 한대 

### 1) Google Colab 환경에서 Konlpy 세팅하기

```python
!apt-get update
!apt-get install g++ openjdk-8-jdk 
!pip install konlpy JPype1-py3
!bash <(curl -s https://raw.githubusercontent.com/konlpy/konlpy/master/scripts/mecab.sh)
```

로 먼저 구글 코랩 환경에서 konlpy를 사용할 수 있게 설치해줍니다.



### 2) 데이터를 읽어와 Tokenization 진행하기

```python
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer()

data = open('./drive/MyDrive/text/k_poem.txt').read()

corpus = data.split("\n")

tokenizer.fit_on_texts(corpus)
total_words = len(tokenizer.word_index) + 1
```

Tokenization은 텐서플로우의 케라스로 진행합니다. 구글 드라이브를 마운트하여 수집한 시 데이터셋을 불러오고 읽어냅니다. 줄 단위로 읽고 corpus에 대해 tokenization을 진행합니다. total_words에 1을 더하는 이유는 Out-Of-Vocabluary(OOV) token을 이용하기 때문에 이를 보충하기 위해서 더합니다.



### 3) n-gram 시퀀스를 생성하고 문장 패딩하기

```python
input_sequences = []
#n-gram 단위로 학습 할 예정.
for line in corpus:
	token_list = tokenizer.texts_to_sequences([line])[0]
	for i in range(1, len(token_list)):
		n_gram_sequence = token_list[:i+1]
		input_sequences.append(n_gram_sequence)

# 가장 긴 줄에 따라 나머지 줄들도 맞춰서 길이를 패딩함
max_sequence_len = max([len(x) for x in input_sequences])
input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))

# create predictors and label
# 예측에 사용되는 시퀀스와 라벨을 설정. 맨 마지막 단어.
xs, labels = input_sequences[:,:-1],input_sequences[:,-1]
ys = tf.keras.utils.to_categorical(labels, num_classes=total_words)
```

읽어들인 데이터를 학습 데이터로 만들기 위한 과정입니다. 특정 단어가 등장하면 어떠한 단어가 다음에 나올 것인지 예측할 수 있도록 n-gram의 형태로 시퀀스들을 만듭니다. 그리고 가장 긴 문장을 기준으로 나머지 시퀀스들은 앞을 0으로 패딩합니다.

또한 이렇게 만들어진 시퀀스들을 쪼개어 예측에 사용될 시퀀스와 라벨을 설정합니다. 



### 4) Training Model을 만들어 훈련시키기

```python
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import numpy as np

model = Sequential()
model.add(Embedding(total_words, 100, input_length=max_sequence_len-1))
# 150개의 unit을 갖고 있는 LSTM layer를 추가
model.add(Bidirectional(LSTM(150)))
model.add(Dense(total_words, activation='softmax'))
adam = Adam(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
history = model.fit(xs, ys, epochs=100, verbose=1)
```

양방향 LSTM 모델을 사용하여 네트워크를 구성합니다. Optimizer로는 Adam, 활성함수는 softmax, 손실함수로는 categorical cross entropy를 적용합니다. epoch수는 100, learning rate는 0.01로 설정합니다.

모델에 앞에서 정의한 xs와 ys를 피팅하여 training을 진행합니다.



![img](data:image/png;base64,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)

모델을 fit하여 나온 결과입니다. accuracy는 최종적으로 0.93정도의 accuracy를 보여줍니다.

### 5) 훈련된 모델을 적용하여 seed text에 대한 다음 단어들 예측하기

```python
seed_text = "날이 밝습니다"
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
날이 밝습니다 있었다 짖는다 밤에 살자 출출이 오면 작은 앞에 아배 앞에는 왕사발에 아들 앞에는 새끼사발에 그득히 사리워오는 것이다 목수네 가까이 싶다고 생각 거외다 언제나 한 방에 들어서 쥔을 붙이없다 뒤뜰에 누워 본다 주저앉어서 나는 가난한 아버지를 가진 것과 것은 무엇인가 흰 홍동이 옷의 슬픔이며 하며 이는 당나귀 타고 것같이 이 이
```

seed text로 "날이 밝습니다"를 입력하고 후의 50개의 단어를 출력하게 하였습니다. 말이 안되는 듯 하면서도 읽혀지기는 하는 시가 완성되었습니다.

### 6) Konlpy를 활용하여 시에 등장하는 명사를 추출하여 제목 정하기 / 시의 형태로 출력하기

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

앞서 세팅해놓은 konlpy를 활용해보려 합니다. Mecab을 사용하여 만들어진 시에 대해 품사를 태깅하고 명사를 추출해봅니다.

```
['날', '밤', '앞', '아배', '앞', '왕', '사발', '아들', '앞', '새끼', '사발', '사리', '워', '목수', '생각', '쥔', '붙이', '뒤', '뜰', '가난', '아버지', '홍동', '옷', '슬픔', '당나귀']
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

최종적으로 시의 형태로 작품을 출력하기 위해 파이썬의 라이브러리들을 활용해봅니다. 한 줄에 30글자(단어 단위로 끊김)를 기준으로 다음 줄로 넘어가게 하고, 앞서 추출한 명사들 중 하나를 랜덤으로 제목으로 정하여 보여지게 합니다. 

```
슬픔


날이 밝습니다 있었다 짖는다 밤에 살자 출출이 오면
작은 앞에 아배 앞에는 왕사발에 아들 앞에는 새끼사발에
그득히 사리워오는 것이다 목수네 가까이 싶다고 생각
거외다 언제나 한 방에 들어서 쥔을 붙이없다 뒤뜰에
누워 본다 주저앉어서 나는 가난한 아버지를 가진 것과
것은 무엇인가 흰 홍동이 옷의 슬픔이며 하며 이는
당나귀 타고 것같이 이 이
```



소소한 조작이지만 보기에는 왠지 그럴듯한 시가 완성되었습니다. (완)



## 참고

- TensorFlow - Natural Language Processing (NLP) Zero to Hero 中

  - [Training an AI to create poetry (NLP Zero to Hero - Part 6) - YouTube](https://www.youtube.com/watch?v=ZMudJXhsUpY&list=PLQY2H8rRoyvzDbLUZkbudP-MFQZwNmU4S&index=6)

  - [https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/TensorFlow%20In%20Practice/Course%203%20-%20NLP/Course%203%20-%20Week%204%20-%20Lesson%202%20-%20Notebook.ipynb](https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/TensorFlow In Practice/Course 3 - NLP/Course 3 - Week 4 - Lesson 2 - Notebook.ipynb)

- [07) 케라스(Keras) 훑어보기 - 딥 러닝을 이용한 자연어 처리 입문 (wikidocs.net)](https://wikidocs.net/32105)
