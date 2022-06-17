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



![img](https://drive.google.com/uc?id=1m9Mbc4OUORwL8fkqsx__SotP0g9WmvdH)

모델을 fit하여 나온 결과입니다. 최종적으로 0.93정도의 accuracy를 보여줍니다.

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

- [07) 케라스(Keras) 훑어보기 - 딥 러닝을 이용한 자연어 처리 입문 (wikidocs.net)](https://wikidocs.net/32105)
