#imports
from flask import Flask, render_template, request
import numpy as np
import nltk
import string
import random
import sklearn

app = Flask(__name__)



@app.route("/")
def index():
    return render_template("index.html")

@app.route("/get",methods=['POST','GET'])
#function for the bot response
def get():
    #if request.method == 'POST':
        user_response =(request.form['msg'])
        #return str(englishBot.get_response(userText))
        f=open('chatbot.txt',errors='ignore')
        raw_doc=f.read()
        raw_doc=raw_doc.lower()
        nltk.download('punkt')
        nltk.download('wordnet')
        sent_tokens=nltk.sent_tokenize(raw_doc)
        word_tokens=nltk.word_tokenize(raw_doc)
            
        lemmer=nltk.stem.WordNetLemmatizer()
        def LemTokens(tokens):
          return [lemmer.lemmatize(token) for token in tokens]
        remove_punct_dict=dict((ord(punct),None) for punct in string.punctuation)
        def LemNormalize(text):
          return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))
        
        GREET_INPUTS=("hello",'hi',"greetings","sup","what's up","hey")
        GREET_RESPONSES=["hi","hey","*nods*","hi there","hello","I am glad ! You are talking to me"]
        def greet(sentence):
          for word in sentence.split():
            if word.lower() in GREET_INPUTS:
              return random.choice(GREET_RESPONSES)
          
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        
        def response(user_response):
          robo1_response=''
          TfidfVec=TfidfVectorizer(tokenizer=LemNormalize,stop_words='english')
          tfidf=TfidfVec.fit_transform(sent_tokens)
          vals=cosine_similarity(tfidf[-1],tfidf)
          idx=vals.argsort()[0][-2]
          flat=vals.flatten()
          flat.sort()
          req_tfidf=flat[-2]
          if (req_tfidf==0):
            robo1_response=robo1_response+"I am sorry ! I dont understand you"
            return robo1_response
          else:
            robo1_response=robo1_response+sent_tokens[idx]
            return robo1_response
        
        flag=True
        #return render_template("Bot: My name is Stark. Lets have a conversation! .ALSO if you want to exit just type ,Bye!")
        while(flag==True):
          #user_response=input()
          user_response=user_response.lower()
          if(user_response!='bye'):
            if(user_response=='thanks' or user_response=='thank you'):
              flag=False
              return render_template('index.html',text="Bot: You are welcome..")
            else:
              if (greet(user_response)!=None):
                  res="Bot: "+greet(user_response)
                  return render_template('index.html',text=res)
              else:
                sent_tokens.append(user_response)
                word_tokens=word_tokens+nltk.word_tokenize(user_response)
                final_words=list(set(word_tokens))
                #return render_template('index1.html',"BOT: ",end="")
                res1=response(user_response)
                sent_tokens.remove(user_response)
                return render_template('index.html',text=res1)
                
          else:
            flag=False
            return render_template('index.html',text="Bot: Goodbye! TAKE care")

if __name__ == "__main__":
    app.run()