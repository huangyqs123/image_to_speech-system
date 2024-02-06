from textblob import TextBlob
import spacy
import contextualSpellCheck
import re

def connect_sentence(text):
    lines = text.split('\n')
    sentence = [None]*len(lines)
    j=0
    end_sentence = False
    connect_word = False
    for i in range (len(lines)):
        if lines[i].endswith(('.', ',','!','?')) == False and len(lines[i])>25: #if sentence is not ending
            end_sentence = False
    
            if sentence[j]==None:#previous sentence must be ended
                if lines[i].endswith('-') ==True:
                    lines[i]=lines[i][:-1]
                    connect_word = True
                sentence[j]=lines[i]
            else:
                if connect_word ==True and lines[i].endswith('-') ==True:#previous and current sentence needs to connect word
                    lines[i]=lines[i][:-1]
                    sentence[j]+=lines[i]
                elif connect_word ==True and lines[i].endswith('-') ==False: #previous sentence needs to connect word, current not
                    sentence[j]+=lines[i]
                    connect_word = False   
                elif connect_word ==False and lines[i].endswith('-') ==True: #current sentence needs to connect word, previous not
                    lines[i]=lines[i][:-1]
                    sentence[j]+=' '+lines[i] #connect to previous sentence
                    connect_word = True
                else: #both not
                    sentence[j]+=' '+lines[i] #connect to previous sentence
                    connect_word = False
                  
        else: #sentence is ended
            if end_sentence ==False: #previous not ended
                if connect_word ==True: #connect to previous sentence
                    sentence[j]+=lines[i]
                else:
                    if sentence[j]==None:
                        sentence[j]=lines[i]
                    else:
                        sentence[j]+=' '+lines[i]    
                end_sentence = True
                j+=1
            else: #previous ended
                sentence[j]=lines[i]
                j+=1
                end_sentence = True
            connect_word = False 
    result = [x for x in sentence if x is not None]
    text = '\n'.join(result)
    return text

def single_spell_check(text):
    tb = TextBlob(text)
    result = tb.correct()
    result = str(result)
    return result

def contextual_spell_check(text):
    nlp = spacy.load("en_core_web_sm") ##need 'python -m spacy download en_core_web_sm'
    contextualSpellCheck.add_to_pipe(nlp)
    doc = nlp(text)
    numError = len(doc._.suggestions_spellCheck)
    # print(doc._.suggestions_spellCheck)
    result = doc._.outcome_spellCheck
    return result

def detect_accuracy(text):
    if len(text) ==0:
        return 0
    
    pattern = r'\b\w+\b'  # Regular expressions to match English words
    matches = re.findall(pattern, text)
    num_words = len(matches)
    
    nlp = spacy.load("en_core_web_sm")
    contextualSpellCheck.add_to_pipe(nlp)
    doc = nlp(text)
    mistake = len(doc._.suggestions_spellCheck)
    accuracy = 1-len(doc._.suggestions_spellCheck)/num_words
    
    # print('num Words:'+str(num_words))
    # print('mistake:'+str(len(doc._.suggestions_spellCheck)))
    # print('accuracy:'+str(1-len(doc._.suggestions_spellCheck)/num_words))
    return accuracy
    


    
    