import codecs
import random
import matplotlib.pyplot as plt
from collections import Counter
from nltk import ngrams
import seaborn as sns

with codecs.open('europarl-v7.cs-en.cs', 'r', encoding='utf-8', errors='ignore') as cs_file:
    czech_sentences = cs_file.readlines()

with codecs.open('europarl-v7.cs-en.en', 'r', encoding='utf-8', errors='ignore') as en_file:
    english_sentences = en_file.readlines()

# selecting 10% of the data
data_sample_size = int(0.1 * len(czech_sentences))
random.seed(42)
data_sample_indices = random.sample(range(len(czech_sentences)), data_sample_size)
czech_sentences_sample = [czech_sentences[i] for i in data_sample_indices]
english_sentences_sample = [english_sentences[i] for i in data_sample_indices]

length_differences = [len(en_sent.split()) - len(cs_sent.split()) for en_sent, cs_sent in zip(english_sentences_sample, czech_sentences_sample)]

avg_length_difference = sum(length_differences) / len(length_differences)
max_length_difference = max(length_differences)
min_length_difference = min(length_differences)

plt.figure(figsize=(10, 6))

# Histogram of length differences
plt.subplot(1, 2, 1)
plt.hist(length_differences, bins=30)
plt.xlabel('Length Difference (English - Czech)')
plt.ylabel('Frequency')
plt.title('Distribution of Sentence Length Differences')

# Box plot of length differences
plt.subplot(1, 2, 2)
plt.boxplot(length_differences, vert=False)
plt.xlabel('Length Difference (English - Czech)')
plt.title('Box Plot of Sentence Length Differences')

plt.tight_layout()
plt.show()

# Most frequent words
czech_words = ' '.join(czech_sentences_sample).split()
english_words = ' '.join(english_sentences_sample).split()

czech_word_freq = Counter(czech_words)
english_word_freq = Counter(english_words)

most_common_czech_words = czech_word_freq.most_common(10)
most_common_english_words = english_word_freq.most_common(10)

# N-gram analysis
n = 2  # Change the value of n for different n-gram analysis
czech_ngrams = list(ngrams(czech_words, n))
english_ngrams = list(ngrams(english_words, n))

czech_ngram_freq = Counter(czech_ngrams)
english_ngram_freq = Counter(english_ngrams)

most_common_czech_ngrams = czech_ngram_freq.most_common(10)
most_common_english_ngrams = english_ngram_freq.most_common(10)

print("\nMost Common Czech Words:")
for word, count in most_common_czech_words:
    print(f"{word}: {count}")

print("\nMost Common English Words:")
for word, count in most_common_english_words:
    print(f"{word}: {count}")

print("\nMost Common Czech " + str(n) + "-grams:")
for ngram, count in most_common_czech_ngrams:
    print(f"{' '.join(ngram)}: {count}")

print("\nMost Common English " + str(n) + "-grams:")
for ngram, count in most_common_english_ngrams:
    print(f"{' '.join(ngram)}: {count}")

# TODO
# Some other insights maybe?

