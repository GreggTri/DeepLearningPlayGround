import nltk
#nltk.download('punkt')
#nltk.download('wordnet')
import torchtext
glove = torchtext.vocab.GloVe(name="6B", dim=100)
#glove.stoi.__setitem__('<PAD>', 0)
#glove.stoi.__setitem__('<UNK>', 1)
#glove.itos.__setitem__(0, '<PAD>')
#glove.itos.__setitem__(1, '<UNK>')
import random
import csv
from nltk.corpus import wordnet

#still have to test and make sure works
def augment_sentence(sentence, ex_query, ex_price, ex_color):
  # Tokenize the sentence into words
  words = nltk.word_tokenize(sentence)
  
  # Replace each word with a synonym if one exists
  for i, word in enumerate(words):
    synonyms = wordnet.synsets(word)
    if synonyms and words[i] != ex_query and words[i]!= ex_price and words[i]!= ex_color:
      # Choose a random synonym
      synonym = synonyms[0].lemmas()[0].name()
      words[i] = synonym
      
  # Rejoin the words into a new sentence
  augmented_sentence = ' '.join(words)
  return augmented_sentence

def shuffle_sentence(sentence):
  # Tokenize the sentence into words
  words = nltk.word_tokenize(sentence)
  
  # Shuffle the words
  random.shuffle(words)
  
  # Rejoin the words into a shuffled sentence
  shuffled_sentence = ' '.join(words)
  return shuffled_sentence

def add_random_word(sentence):
    words = nltk.word_tokenize(sentence)
    
    word_index = random.randint(0, len(words) - 1)
    
    words.insert(word_index, random.choice(glove.itos))
    modified_sentence = ' '.join(words)
    return modified_sentence

with open('Data/MQdataset.csv', 'r') as file:
    reader = csv.reader(file)
    next(reader)
    
    with open('Data/AUGMQdataBS.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)    
        writer.writerow(['sentence', 'query0', 'query1', 'query2', 'query3', 'color'])
        for row in reader:
            count = 0
            writer.writerow([row[0], row[1], row[2], row[3], row[4], row[5]])
            
            print(row[0])
                
            aug_sen = shuffle_sentence(row[0])
            aug_sen = add_random_word(aug_sen)
            
            writer.writerow([aug_sen, row[1], row[2], row[3], row[4], row[5]])
            
            aug_sen = shuffle_sentence(aug_sen)
            aug_sen = add_random_word(aug_sen)
            
            writer.writerow([aug_sen, row[1], row[2], row[3], row[4], row[5]])
            #print(aug_sen)
            count += 1
            

