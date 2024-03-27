!pip install translate
from translate import Translator
import pandas as pd

def translate_th(df,cl) :
  for i in cl :
    translator = Translator(to_lang = 'en')
    df['en'] = df[cl].apply(translator.translate)
    return df

translate_th(product,'th')
