import pandas as pd
import string
import re
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')



from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt


df1 = pd.read_csv(r'D:/AI/LangDetect/Datasets/Language Detection.csv')
df2 = pd.read_csv(r'D:/AI/LangDetect/Datasets/hindi.csv')


df1

df2

df = df1.append(df2,ignore_index=True)

df

df.info()

df.Language.value_counts()


df[df.Language == 'Russian'].sample(2)

df[df.Language == 'Malayalam'].sample(2)


df[df.Language == 'Arabic'].sample(2)


df[df.Language == 'Tamil'].sample(2)


df[df.Language == 'Kannada'].sample(2)


df[df.Language == 'Hindi'].sample(2)



def removeSymbolsAndNumbers(text):        
        text = re.sub(r'[{}]'.format(string.punctuation), '', text)
        text = re.sub(r'\d+', '', text)
        text = re.sub(r'[@]', '', text)

        return text.lower()
    
    
    
def removeEnglishLetters(text):        
        text = re.sub(r'[a-zA-Z]+', '', text)
        return text.lower()
    
X0 = df.apply(lambda x: removeEnglishLetters(x.Text) if x.Language in ['Russian','Malyalam','Hindi','Kannada','Tamil','Arabic']  else x.Text, axis = 1)
X0 

X1 = X0.apply(removeSymbolsAndNumbers)
X1


y = df['Language']


x_train, x_test, y_train, y_test = train_test_split(X1,y, random_state=42)



vectorizer = TfidfVectorizer(ngram_range=(1,3), analyzer='char')



model = pipeline.Pipeline([
    ('vectorizer', vectorizer),
    ('clf', LogisticRegression())
])


model.fit(x_train,y_train)


y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test,y_pred)
cm = confusion_matrix(y_test,y_pred)


print("Accuracy is :",accuracy)


print(classification_report(y_test,y_pred))


plt.figure(figsize=(15,10))
sns.heatmap(cm, annot = True)
plt.show()



def predict(text):
    lang = model.predict([text])
    print('The Detected Language is ',lang[0])
    
    
    
#Tamil
predict("புத்தகம்")
#English
predict("Language Detection")
#Kannada
predict("	ಮಾಹಿತಿ")
# French
predict("VÉRIFICATION DU MODÈLE DE DÉTECTION DE LA LANGUE")
# Arabic
predict("توففحص نموذج الكشف عن اللغة")
#Spanish
predict("VERIFICACIÓN DEL MODELO DE DETECCIÓN DE IDIOMAS")
# Malayalam
predict("ലാംഗ്വേജ് ഡിറ്റക്ഷൻ മോഡൽ ചെക്ക്")
# Russian
predict("ПРОВЕРКА МОДЕЛИ ОПРЕДЕЛЕНИЯ ЯЗЫКА")
# Hindi
predict('प्रयोगशाला')
# Hindi
predict(' boyit9h एनालिटिक्स alhg//serog 90980879809 bguytfivb ahgseporiga प्रदान करता है')
#Turkish
predict("Tak")
#Dutch
predict("Goedemorgen")
#Greek
predict("ψυχή")
#Italian
predict(" Grazie")
#Portugese
predict("Obrigada/o")
#Sweedish
predict("Varsågod")
#Danish
predict("Rødovre & Hvidovre")
    