import codecs
import random
import re

with codecs.open('europarl-v7.cs-en.cs', 'r', encoding='utf-8', errors='ignore') as cs_file:
    czech_sentences = cs_file.readlines()

with codecs.open('europarl-v7.cs-en.en', 'r', encoding='utf-8', errors='ignore') as en_file:
    english_sentences = en_file.readlines()

data = list(zip(czech_sentences, english_sentences))
# The preprocessing steps I have choosen unter are:
#   1.remove lines with XML-Tags (starting with "<")
#   2.strip empty lines and their correspondences
#   3.lowercase the text
# If you have some new ideas, just code and add in this list.
data_processed = []
xml_tag_pattern = re.compile(r'<[^>]+>')

for czech_sent, english_sent in data:
    czech_sent = czech_sent.strip()
    english_sent = english_sent.strip()

    if czech_sent and english_sent and not xml_tag_pattern.match(czech_sent):
        czech_sent = czech_sent.lower()
        english_sent = english_sent.lower()

        data_processed.append((czech_sent, english_sent))

data_sample_size = int(0.00001 * len(data_processed))
random.seed(8)
data_sample = random.sample(data_processed, data_sample_size)

for czech_sent, english_sent in data_sample:
    print("Czech:", czech_sent)
    print("English:", english_sent)
    print("----------------------")