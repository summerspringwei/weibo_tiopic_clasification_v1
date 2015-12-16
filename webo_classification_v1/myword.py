# encoding: UTF-8

import nlpir as nlp

'''
author: ChunweiXia of Tianjin University
这个文件封装了单词类、句子类和将训练集合转换成为词典的工具类
'''


class MyWord(object):
    """
    单词类，封装单词、属性、一共出现次数、在正面话题、0活体、负面话题出现次数，进而计算idf值
    """
    def __init__(self, text, attr):
        self.text = text
        self.attr = attr
        self.total_count = 0
        self.positive_count = 0
        self.negative_count = 0
        self.zero_count = 0

    def get_positive_idf(self):
        return self.positive_count / self.total_count

    def get_zero_idf(self):
        return self.zero_count / self.total_count

    def get_negative_idf(self):
        return self.negative_count / self.total_count

    def inc_zero_count(self):
        self.zero_count += 1
        self.total_count += 1

    def inc_positive_count(self):
        self.positive_count += 1
        self.total_count += 1

    def inc_negative_count(self):
        self.negative_count += 1
        self.total_count += 1

    def word2vector(self):
        return [self.get_positive_idf(), self.get_zero_idf(), self.get_negative_idf()]


class MyDictionary:
    """
    通过读取微博话题文件，调用ICTNLP库形成词典
    """
    def __init__(self, file_path):
        self.file_path = file_path
        self.mdict = {}

    def parse_file(self):
        """
        把训练文件的句子分解成词语，并打上标签
        :return:
        """
        f = open(self.file_path, 'r')
        line = f.readline()
        label = self.get_label(line)
        text = self.get_text(line)
        for token in nlp.seg(text):
            if self.word_filter(token[1]):
                continue
            if token[0] in self.mdict.keys():
                mword = self.mdict.get(token[0])
                inc_operator = {"+1": mword.inc_positive_count,
                                "0": mword.inc_zero_count,
                                "-1": mword.inc_negative_count}
                inc_operator.get(label)()
            else:
                mword = MyWord(token[0], token[1])
                self.mdict[token[0]] = mword

    @staticmethod
    def word_filter(attr):
        """
        如果属性值在过滤集合中，返回True
        :param attr:
        :return:
        """
        attr_list = {"t", "f", "m", "x", "w"}
        filter_attr_set = set(attr_list)
        if attr[0] in filter_attr_set:
            return True
        return False

    @staticmethod
    def get_label(line):
        """
        获取句子前面的标签
        """
        if len(line) < 2:
            return "0"
        return line[0:2].strip().lstrip().rstrip()

    @staticmethod
    def get_text(line):
        """
        :param line:
        :return: 去掉前后空格的句子
        """
        if len(line) < 6:
            return ""
        return line[6:len(line)].strip().lstrip().rstrip()


class MySentence(object):
    """
      封装句子，将句子转换为向量
    """
    def __init__(self, sentence, mdictionary):
        self.sentence = sentence
        self.mdictionary = mdictionary
        self.positive_word_count = 0
        self.zero_word_count = 0
        self.negative_word_count = 0
        self.word_list = []

    def inc_positive_word_count(self):
        self.positive_word_count += 1

    def inc_zero_word_count(self):
        self.zero_word_count += 1

    def inc_negative_count(self):
        self.negative_word_count += 1

    def get_total_word_count(self):
        total = self.positive_word_count + self.zero_word_count + self.negative_word_count
        if total > 1:
            return total
        else:
            return 1

    def sentence2vector(self):
        """
        返回本句子转换成的向量
        :return:mlist
        """
        mlist = [self.positive_word_count / self.get_total_word_count(),
                 self.zero_word_count / self.get_total_word_count(),
                 self.negative_word_count / self.get_total_word_count()]
        for word in self.word_list:
            mlist.extend(word.word2vector())
        return mlist

    def parse_sentence(self):
        feature_attr_list = ["a", "v", "z", "d", "e"]
        feature_word_count = 0
        feature_word_num = 3
        token_list = nlp.seg(self.sentence)
        for token in token_list:
            # 获取整个句子每个单词的正向、0、负向之和
            if token[0] in self.mdictionary:
                mword = self.mdictionary.get(token[0])
                self.positive_word_count += mword.positive_count
                self.negative_word_count += mword.netative_count
                self.zero_word_count += mword.zero_count
                # 如果单词有特征属性，则加入句子的单词列表
                if mword.attr[0] in feature_attr_list and feature_word_count < feature_word_num:
                    self.word_list.append(mword)
                    feature_word_count += 1
        # 如果特征单词不足三个
        if feature_word_count < feature_word_num:
            # 放入句子前两个
            if len(token_list) > 2:
                for i in range(feature_word_num-len(self.word_list)):
                    self.word_list.append(self.mdictionary[token_list[i]])
            else:
                for i in range(feature_word_num-len(self.word_list)):
                    self.word_list.append(self.mdictionary[token_list[0]])


p = "-1 今天鬼节。为各路鬼神祈福。今天说邪门还真有点。状况百出。巧合吧。从外面回来又坐在床前发呆了。什么都不想做。赶紧收拾好洗漱完躺下休息吧。要保护好身体。"
for t in nlp.Seg(p):
    s = '%s\t%s\t%s' % (t[0],t[1],nlp.translatePOS(t[1]))
    print(s)
