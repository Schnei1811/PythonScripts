import nltk
from nltk.tokenize import sent_tokenize, word_tokenize, PunktSentenceTokenizer          #punkt unsupervised learning algorithm
from nltk.corpus import stopwords, state_union
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import gutenberg
from nltk.corpus import wordnet


# tokenizing - word tokenizers. separates by word. sentence tokenizer. separates by sentence
# lexicon and coporas
# lexicon - words and their meanings
#     investor speak vs regular english speak
#     investor 'bull' - someone who is positive about the market
#     english 'bull' - scary animal you don't want running at you
# corpora - body of text. ex: medical journals, presidential speeches, English language

# stop words: words to eliminate for confusion or uselessness

example_text = 'Hello there, how are you doing today? The weather is great and Python is awesome. The sky is pinkish-blue. You should not eat cardboard'

#print(sent_tokenize(example_text))
#print(word_tokenize(example_text))
#for i in word_tokenize(example_text):
#    print(i)

example_sentence = "This is an example showing off stop word filtration."
stop_words = set(stopwords.words('english'))
#print(stop_words)

words = word_tokenize(example_sentence)

filtered_sentence = [w for w in words if not w in stop_words]
print(filtered_sentence)
# filtered_sentence = []
# for w in words:
#     if w not in stop_words:
#         filtered_sentence.append(w)
#
# print(filtered_sentence)

ps = PorterStemmer()
#example_words = ['python','pythoner','pythoning','pythoned','pythonly']
#for w in example_words:
#    print(ps.stem(w))

new_text = 'It is very important to be pythonly while you are pythoning with python. All pythoners haved pythoned poorly at least once'

words = word_tokenize(new_text)

for w in words:
    print(ps.stem(w))

'''
POS tag list:

CC	coordinating conjunction
CD	cardinal digit
DT	determiner
EX	existential there (like: "there is" ... think of it like "there exists")
FW	foreign word
IN	preposition/subordinating conjunction
JJ	adjective	'big'
JJR	adjective, comparative	'bigger'
JJS	adjective, superlative	'biggest'
LS	list marker	1)
MD	modal	could, will
NN	noun, singular 'desk'
NNS	noun plural	'desks'
NNP	proper noun, singular	'Harrison'
NNPS	proper noun, plural	'Americans'
PDT	predeterminer	'all the kids'
POS	possessive ending	parent's
PRP	personal pronoun	I, he, she
PRP$	possessive pronoun	my, his, hers
RB	adverb	very, silently,
RBR	adverb, comparative	better
RBS	adverb, superlative	best
RP	particle	give up
TO	to	go 'to' the store.
UH	interjection	errrrrrrrm
VB	verb, base form	take
VBD	verb, past tense	took
VBG	verb, gerund/present participle	taking
VBN	verb, past participle	taken
VBP	verb, sing. present, non-3d	take
VBZ	verb, 3rd person sing. present	takes
WDT	wh-determiner	which
WP	wh-pronoun	who, what
WP$	possessive wh-pronoun	whose
WRB	wh-abverb	where, when
'''

train_text = state_union.raw("2005-GWBush.txt")
sample_text = state_union.raw("2006-GWBush.txt")

custom_sent_tokenizer = PunktSentenceTokenizer(sample_text)     #Training tokenizer on provided text

tokenized = custom_sent_tokenizer.tokenize(sample_text)

def process_content():
    try:
        for i in tokenized:
            words = nltk.word_tokenize(i)
            tagged = nltk.pos_tag(words)

            #chunkGram = r'''Chunk: {<RB.?>*<VB.?>*<NNP>+<NN>?}'''          #Regular Expressions. https://pythonprogramming.net/regular-expressions-regex-tutorial-python-3/

            chunkGram = r'''Chunk: {<.*>+}
                                    }<VB.?|IN|DT|TO>+{'''                                   #Chinkin. }{

            chunkParser = nltk.RegexpParser(chunkGram)                      #Chumking. Grouping of things
            chunked = chunkParser.parse(tagged)

            #print(tagged)
            print(chunked)
            chunked.draw()
    except Exception as e:
        print(str(e))

# def process_content():
#     try:
#         for i in tokenized[5:]:
#             words = nltk.word_tokenize(i)
#             tagged = nltk.pos_tag(words)
#             namedEnt = nltk.ne_chunk(tagged, binary=True)
#             namedEnt.draw()
#     except Exception as e:
#         print(str(e))
#
process_content()

#lemmetizing some form of synonym to a word. Default noun. Lemmatizing better then stemming

lemmatizer = WordNetLemmatizer()

print(lemmatizer.lemmatize('cats'))
print(lemmatizer.lemmatize('cacti'))
print(lemmatizer.lemmatize('geese'))
print(lemmatizer.lemmatize('rock'))
print(lemmatizer.lemmatize('python'))
print(lemmatizer.lemmatize('better'))
print(lemmatizer.lemmatize('better',pos='a'))
print(lemmatizer.lemmatize('best',pos='a'))
print(lemmatizer.lemmatize('run'))
print(lemmatizer.lemmatize('run','v'))

#Corpus

sample = gutenberg.raw('bible-kjv.txt')
tok = sent_tokenize(sample)
print(tok[5:15])

syns = wordnet.synsets('program')

#synset
print(syns[0].name())

#just the word
print(syns[0].lemmas()[0].name())

#definition
print(syns[0].definition())

#examples
print(syns[0].examples())

synonyms = []
antonyms = []

for syn in wordnet.synsets('good'):
    for l in syn.lemmas():
        synonyms.append(l.name())
        if l.antonyms():
            antonyms.append(l.antonyms()[0].name())

print(set(synonyms))
print(set(antonyms))

w1 = wordnet.synset('ship.n.01')
w2 = wordnet.synset('boat.n.01')
print(w1.wup_similarity(w2))

w1 = wordnet.synset('ship.n.01')
w2 = wordnet.synset('car.n.01')
print(w1.wup_similarity(w2))

w1 = wordnet.synset('ship.n.01')
w2 = wordnet.synset('cat.n.01')
print(w1.wup_similarity(w2))

w1 = wordnet.synset('ship.n.01')
w2 = wordnet.synset('cactus.n.01')
print(w1.wup_similarity(w2))
