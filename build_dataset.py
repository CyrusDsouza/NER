import os
import mammoth
from bs4 import BeautifulSoup
import re
import contractions
import inflect
import unicodedata
from nltk import word_tokenize, sent_tokenize, pos_tag
from nltk.corpus import stopwords
import csv
import spacy
nlp = spacy.load('en_core_web_sm')
from pickle import load

# try:
#     word_embeddings = load(open("models/numberbatch-en.pkl",'rb'))
#     print("Loaded Word Embeddings")

# except Exception as e:
#     print(e)

class NERDataProcessor(object):

    def __init__(self , important_entities = False, allowed_tags = False):
        self.total_number = 1
        if not important_entities:
            self.important_entities = {}
        else:
            self.important_entities = important_entities

        if not allowed_tags: 
            self.allowed_tags = []
        else:
            self.allowed_tags = allowed_tags

    def _remove_non_ascii(self, words):
        """Remove non-ASCII characters from list of tokenized words"""
        new_words = []
        for word in words:
            new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
            new_words.append(new_word)
        return new_words

    def _to_lowercase(self, words):
        """Convert all characters to lowercase from list of tokenized words"""
        new_words = []
        for word in words:
            new_word = word.lower()
            new_words.append(new_word)
        return new_words

    def _remove_punctuation(self, words):
        """Remove punctuation from list of tokenized words"""
        new_words = []
        for word in words:
            new_word = re.sub(r'[^\w\s]', ' ', word)
            if new_word != '':
                new_words.append(new_word)
        return new_words

    def _replace_numbers(self, words):
        """Replace all interger occurrences in list of tokenized words with textual representation"""
        p = inflect.engine()
        new_words = []
        for word in words:
            if word.isdigit():
                new_word = p.number_to_words(word)
                new_words.append(new_word)
            else:
                new_words.append(word)
        return new_words

    def _remove_stopwords(self, words):
        """Remove stop words from list of tokenized words"""
        new_words = []
        for word in words:
            if word not in stopwords.words('english'):
                new_words.append(word)
        return new_words
        
    def _strip_html(self,text):
        soup = BeautifulSoup(text, "html.parser")
        return soup.get_text()

    def _remove_between_square_brackets(self,text):
        return re.sub(r'\[[^]]*\]', ' ', text)

    def _replace_contractions(self,text):
        """Replace contractions in string of text"""
        return contractions.fix(text)

    def denoise_text(self, text):
        text = self._strip_html(text)
        text = self._remove_between_square_brackets(text)
        text = self._replace_contractions(text)
        return text

    def normalize(self, words):
        words = self._remove_non_ascii(words)
        words = self._to_lowercase(words)
        words = self._remove_punctuation(words)
        words = self._replace_numbers(words)
        # words = self._remove_stopwords(words)

        return words

    def get_POS(self, text):
        sentences = sent_tokenize(text)
        all_meta = []
        for sentnumber,sent in enumerate(sentences,1): 
            tokens = word_tokenize(sent)
            refined_sentence = ' '.join(tokens)
            self.spacy = nlp(unicode(refined_sentence))
            self.spacy_labels =  [(X.text.lower(), X.label_) for X in self.spacy.ents]
            self.spacy_postags = [(token.text, token.tag_) for token in self.spacy]
            self.spacy_phrases = {chunk.text.lower() : (chunk.label_, chunk.root.text) for chunk in self.spacy.noun_chunks}
        

            print("Sentence {}".format(sentnumber))

            if 0:
                print("-"*100)
                print
                print("Tokens----> " , tokens)
                print
                # print("Ranked Phrases---> ", self.ranked_phrases)
                print
                print("Spacy Phrases --> ", self.spacy_phrases)
                print
                print("Spacy NER --->  ", self.spacy_labels)
                print
                print("Spacy POS -->  ", self.spacy_postags)
                print("-"*100)


            tokens.extend([phrase for phrase in self.spacy_phrases])

            tokens = self.normalize(tokens)
            pos_tagged = pos_tag(tokens)
            refined_sentence = ' '.join(tokens).strip()

            # if self.important_entities:

            #     self.tag_entities(refined_sentence)
            # import pdb; pdb.set_trace()

            tags = []
            for word,pos in pos_tagged:
                
                if self.important_entities.get(word,False):
                    tags.append(self.important_entities.get('word','O'))
                
                elif self.spacy_phrases.get(word,False):
                    if self.spacy_phrases.get(word)[1].lower() in self.allowed_tags:
                        tags.append(self.spacy_phrases.get(word)[1])
                    else:
                        tags.append('O')

                else:
                    tags.append('O')
                    
            # tags = ['O' for _ in range(len(pos_tagged))]
            
            meta = [(x[0][0],x[0][1],x[1].lower()) for x in zip(pos_tagged,tags)]

            all_meta.append(meta)

        return all_meta
            
    def create_dataset(self, meta):

        file_exists = os.path.isfile("models/{}".format('contract_ner1.csv'))
        try:
            with open('models/contract_ner1.csv', 'a') as writefile:
                contract_writer = csv.DictWriter(writefile, lineterminator = '\n', fieldnames = ['SentenceNumber', 'Word', 'POS', 'Tag'])
                
                if not file_exists:
                    contract_writer.writeheader()


                for i, row in enumerate(meta,self.total_number): #sentences
                    for wordnumber, (word, pos, tag) in enumerate(row): #word in each sentence
                        if not len(word.strip()): #not a word.
                            continue

                        if self.important_entities:
                            if word in self.important_entities:
                                tag = self.important_entities.get(word,'o')
                            

                        if wordnumber: #FIRST WORD IN THE SENTENCE
                            contract_writer.writerow({"SentenceNumber": "", 
                                                        "Word": word,
                                                        "POS" : pos,
                                                        "Tag" : tag})
                        else:
                            contract_writer.writerow({"SentenceNumber": "Sentences{}".format(i), 
                                                        "Word": word,
                                                        "POS" : pos,
                                                        "Tag" : tag})

                self.total_number += len(meta)

            return True

        except Exception as e: 
            print(e)
            return False


    def process_data(self, directory):

        for f in os.listdir(directory):
            with open('{}/{}'.format(directory,f), 'rb') as flav: 
                print("Processing file ---> {}".format(os.path.basename(f)))
                try:
                    # result = mammoth.convert_to_html(flav)
                    # doc_results = result.value
                    result = mammoth.extract_raw_text(flav)
                    doc_results = result.value
                    results = self.denoise_text(doc_results)
                    meta  = self.get_POS(results)
                    status = self.create_dataset(meta)
                except Exception as e:
                    print(e) 
                    continue

                # messages = doc_results.messages
                

if __name__ == "__main__":

    directory = 'data/'
    
    allowed_tags = ["company", "customer", "services", "party", "services", "agreement", "products", "seller","buyer", "devices", "section", "devices", "it", "terms", "software", "parties", "date", "time", "information"]                     


    np = NERDataProcessor(important_entities = {'hp' : 'company', 'dxc': 'company', 'delaware': 'company'},
                          allowed_tags = allowed_tags )
    np.process_data(directory)