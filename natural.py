import random
import re
import string

from nltk.corpus import twitter_samples
from nltk.tag import pos_tag, pos_tag_sents
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk import FreqDist
from nltk import NaiveBayesClassifier
import jieba
import requests
def fenci(file):
    return twitter_samples.tokenized(file)


def cleaned_list_func(evert_tweet):
    new_text = []
    cixing_list = pos_tag(evert_tweet)
    for word, cixing in cixing_list:
        word = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|(?:[0-9a-fA-F][0-9a-fA-F]))+', '', word)
        word = re.sub('(@[A-Za-z0-9_]+)', '', word)
        if cixing.startswith('NN'):
            pos = 'n'
        elif cixing.startswith('VB'):
            pos = 'v'
        else:
            pos = 'a'
        lemmatizer = WordNetLemmatizer()
        new_word = lemmatizer.lemmatize(word, pos)
        if len(new_word) > 0 and new_word not in string.punctuation and new_word.lower() not in stopwords.words(
                'english'):
            new_text.append(new_word.lower())
    return new_text


def get_all_words(clean_tokens_list):
    for tokens in clean_tokens_list:
        for token in tokens:
            yield token


def get_tweets_for_model(clean_tokens_list, tag):
    li = []
    for every_tweet in clean_tokens_list:
        data_dict = dict([token, True] for token in every_tweet)
        li.append((data_dict, tag))
    return li


def train_model(train_data, test_data):
    from nltk import classify
    from nltk import NaiveBayesClassifier
    model = NaiveBayesClassifier.train(train_data)
    return model


def test(model, test_text):
    from nltk.tokenize import word_tokenize
    custom_tokens = cleaned_list_func(word_tokenize(test_text))
    result = dict([token, True] for token in custom_tokens)
    yuce_res = model.classify(result)
    print('内容：{}预测结果：{}'.format(test_text,yuce_res))

def translate(str):
    data = {
        'doctype': 'json',
        'type': 'AUTO',
        'i': str
    }
    url = "http://fanyi.youdao.com/translate"
    r = requests.get(url, params=data)
    result = r.json()
    x = result['translateResult']

    y = x[0][0]
    print(y)
    y2 = x[0][1]
    print(y2)
    z = y['tgt']
    z2 = y2['tgt']
    i = z + z2
    print(i)
    return i

if __name__ == '__main__':
    po_file_path = 'positive_tweets.json'
    ne_file_path = 'negative_tweets.json'

    positive_tweets = twitter_samples.strings(po_file_path)
    negative_tweets = twitter_samples.strings(ne_file_path)

    po_fenci_res = fenci(po_file_path)
    be_fenci_res = fenci(ne_file_path)
    positive_cleaned_list = []
    negative_cleaned_list = []
    for i in po_fenci_res:
        positive_cleaned = cleaned_list_func(i)
        positive_cleaned_list.append(positive_cleaned)
    for j in be_fenci_res:
        negative_cleaned = cleaned_list_func(j)
        negative_cleaned_list.append(negative_cleaned)
    po_for_model = get_tweets_for_model(positive_cleaned_list, 'Positive')
    ne_for_model = get_tweets_for_model(negative_cleaned_list, 'Negative')

    model_data = po_for_model + ne_for_model
    random.shuffle(model_data)

    train_data = model_data[:7000]
    test_data = model_data[7000:]

    model = train_model(train_data, test_data)
    test_string = translate("说是618大促，但很多东西都是促销前几天，售价暗地涨起来，对于消费者来说，根本没有什么优惠！对于我们消费者来说，就是掉进了美丽的陷阱。")
    test_list = [test_string]

    for i in test_list:
        test(model, i)