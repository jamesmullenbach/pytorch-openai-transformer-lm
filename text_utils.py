import re
import ftfy
import json
import spacy

from tqdm import tqdm

def get_pairs(word):
    """
    Return set of symbol pairs in a word.
    word is represented as tuple of symbols (symbols being variable-length strings)
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs

def text_standardize(text):
    """
    fixes some issues the spacy tokenizer had on books corpus
    also does some whitespace standardization
    """
    text = text.replace('—', '-')
    text = text.replace('–', '-')
    text = text.replace('―', '-')
    text = text.replace('…', '...')
    text = text.replace('´', "'")
    text = re.sub(r'''(-+|~+|!+|"+|;+|\?+|\++|,+|\)+|\(+|\\+|\/+|\*+|\[+|\]+|}+|{+|\|+|_+)''', r' \1 ', text)
    text = re.sub(r'\s*\n\s*', ' \n ', text)
    text = re.sub(r'[^\S\n]+', ' ', text)
    return text.strip()

class TextEncoder(object):
    """
    mostly a wrapper for a public python bpe tokenizer
    """

    def __init__(self, encoder_path, bpe_path):
        self.nlp = spacy.load('en', disable=['parser', 'tagger', 'ner', 'textcat'])
        self.encoder = json.load(open(encoder_path))
        self.decoder = {v:k for k,v in self.encoder.items()}
        merges = open(bpe_path, encoding='utf-8').read().split('\n')[1:-1]
        merges = [tuple(merge.split()) for merge in merges]
        self.bpe_ranks = dict(zip(merges, range(len(merges))))
        self.cache = {}
        #special tokens
        self.encoder['_start_'] = len(self.encoder)
        self.encoder['_delimiter_'] = len(self.encoder)
        self.encoder['_classify_'] = len(self.encoder)

    def bpe(self, token):
        word = tuple(token[:-1]) + ( token[-1] + '</w>',)
        if token in self.cache:
            return self.cache[token]
        pairs = get_pairs(word)

        if not pairs:
            return token+'</w>'

        while True:
            bigram = min(pairs, key = lambda pair: self.bpe_ranks.get(pair, float('inf')))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word)-1 and word[i+1] == second:
                    new_word.append(first+second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = ' '.join(word)
        if word == '\n  </w>':
            word = '\n</w>'
        self.cache[token] = word
        return word

    def encode(self, texts, verbose=True):
        texts_tokens = []
        for text in tqdm(texts, ncols=80, leave=False, disable=not verbose):
            text = self.nlp(text_standardize(ftfy.fix_text(text)))
            text_tokens = []
            for token in text:
                text_tokens.extend([self.encoder.get(t, 0) for t in self.bpe(token.text.lower()).split(' ')])
            texts_tokens.append(text_tokens)
        return texts_tokens

class TextSelectIndexEncoder(TextEncoder):

    def append_or_replace(self, loc, ix, val):
        if loc[ix] == -1:
            loc[ix] = val
        elif type(loc[ix]) == tuple:
            loc[ix] = (*loc[ix], val)
        else:
            loc[ix] = (loc[ix], val)
        return loc

    def encode(self, texts, triples=None, hide_words=False, verbose=True):
        texts_tokens = []
        locs = []
        for text, triple in tqdm(zip(texts, triples), ncols=80, leave=False, disable=not verbose):
            text = self.nlp(text_standardize(ftfy.fix_text(text)))
            text_tokens = []
            loc = [-1,-1,-1]
            if len(triple[0].split()) > 1:
                multi1, multi2 = triple[0].split()
                search_multi = True
                start_of_multi = False
            else:
                search_multi = False
            for token in text:
                for t in self.bpe(token.text.lower()).split(' '):
                    stem = t.split('</w>')[0]
                    if search_multi:
                        if stem == multi1:
                            start_of_multi = True
                        elif start_of_multi and stem == multi2:
                            loc = self.append_or_replace(loc, 0, (len(text_tokens)-1, len(text_tokens)))
                    elif stem == triple[0]:
                        loc = self.append_or_replace(loc, 0, len(text_tokens))
                    elif stem == triple[1]:
                        loc[1] = len(text_tokens)
                    elif stem == triple[2]:
                        loc[2] = len(text_tokens)
                    text_tokens.append(self.encoder.get(t,0))
            texts_tokens.append(text_tokens)
            locs.append(loc)
        return texts_tokens, locs

class TextHideWordsEncoder(TextEncoder):

    def __init__(self, encoder_path, bpe_path):
        super(TextHideWordsEncoder, self).__init__(encoder_path, bpe_path)
        #add special tokens
        self.encoder['_WHOLE_'] = len(self.encoder)
        self.encoder['_PART_'] = len(self.encoder)
        self.encoder['_ADJECTIVE_'] = len(self.encoder)

    def encode(self, texts, triples=None, verbose=True):
        texts_tokens = []
        for text, triple in tqdm(zip(texts, triples), ncols=80, leave=False, disable=not verbose):
            text = self.nlp(text_standardize(ftfy.fix_text(text)))
            text_tokens = []
            for token in text:
                for t in self.bpe(token.text.lower()).split(' '):
                    stem = t.split('</w>')[0]
                    if stem == triple[0]:
                        tok = self.encoder['_WHOLE_']
                    elif stem == triple[1]:
                        tok = self.encoder['_PART_']
                    elif stem == triple[2]:
                        tok = self.encoder['_ADJECTIVE_']
                    else:
                        tok = self.encoder.get(t,0)
                    text_tokens.append(tok)
            texts_tokens.append(text_tokens)
        return texts_tokens


