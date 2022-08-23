```python
# Set up
import pandas as pd
df = pd.read_csv('data/EAR_english_EMA_20220817.csv', encoding ='utf-8-sig' )

# Group by newid
df = df.fillna('').groupby(['newid'])['transcription'].apply('\n'.join).reset_index()
df['row_num'] = df.index+1

# Clean up transcription
df['transcription'] = df['transcription'].str.replace('[','', regex=True)
df['transcription'] = df['transcription'].str.replace(']','', regex=True)

# Below removes ASCII ellipsis
df['transcription'] = df['transcription'].str.replace('\u2026','', regex=True)
df.head()

# Total Number of Workds in text
df['Num_words'] = df['transcription'].apply(lambda x:len(str(x).split()))
df.head()
```

## NLTK


```python
import nltk
from nltk import word_tokenize, pos_tag, pos_tag_sents
from collections import Counter
from itertools import chain

# Tag tokens, text, POS
tok_and_tag = lambda x: pos_tag(word_tokenize(x))

def word_tokenize(sentence):
    return sentence.split()

df['tokens'] = df['transcription'].astype(str).apply(word_tokenize)
df['text']=[nltk.Text(tokens) for tokens in zip(df['tokens'])]
df['POS'] = df['transcription'].apply(tok_and_tag)

# Count Nouns

def add_pos_with_zero_counts(counter, keys_to_add):
    for k in keys_to_add:
        counter[k] = counter.get(k, 0)
    return counter

def NounCounter(x):
    nouns = []
    for (word, pos) in x:
        if pos.startswith("NN"):
            nouns.append(word)
    return nouns

# Create subset of dataset where Num_words does not equal zero 
df_sub=df[df["Num_words"]!=0]

# Tag nouns in pos, count number of nouns
df_sub['pos_counts'] = df_sub['POS'].apply(lambda x: Counter(list(zip(*x))[1]))
df_sub['pos_counts_with_zero'] = df_sub['pos_counts'].apply(lambda x: add_pos_with_zero_counts(x, possible_tags))
df_sub['sent_vector'] = df_sub['pos_counts_with_zero'].apply(lambda x: [count for tag, count in sorted(x.most_common())])
df_sub["nouns"] = df_sub["POS"].apply(NounCounter)
df_sub["noun_count"] = df_sub["nouns"].str.len()

# Keep only newid, transcription, row_num, Num_words, tokens, text, POS, noun_count from df_sub
df_sub_clean = pd.DataFrame(df_sub[["newid", "transcription", "row_num", "Num_words", "tokens", "text", "POS", "noun_count"]])

# merge df_sub_clean with df
df2=pd.merge(df, df_sub_clean, on=["newid", "row_num"], how="left")

# rename columns
df2 = df2.rename(columns={"transcription_x": "transcription",'Num_words_x':'Num_words','tokens_x': 'tokens', 'text_x':'text', 'POS_x':'POS'})

# delete unnecessary columns
del df2['transcription_y'], df2['Num_words_y'], df2['tokens_y'], df2['text_y'], df2['POS_y']

# replace noun_count NaN to 0
df2['noun_count'] = df2['noun_count'].fillna(0).astype(int)
df2.head()
```

## Usage of Uncommon Words (Average Frequency of Nouns)


```python
# Add average frequency of nouns 
df2["avg_freq_nouns"] = df2["noun_count"]/df2["Num_words"]
df2['avg_freq_nouns'] = df2['avg_freq_nouns'].fillna(0)
df2.head()
```

## Grammatical Complexity (Clauses per sentence)

### Analyzing Grammatical Complexity uses TAASSC
### https://www.linguisticanalysistools.org/taassc.html
### Each transcript needs to be exported as a txt file to feed into TAASSC


```python
# drop instances with no transcription
new_df2=df2[(df2['Num_words']!=0) & (df2['row_num']!=460) & (df2['row_num']!=2765) & (df2['row_num']!=1854) &(df2['row_num']!=4129)]

# Note that leaving empty spaces 
new_df2['transcription'] = new_df2['transcription'].str.replace('\n','\n.', regex=True)

# export every line of csv to txt file
import csv
with open ('data\EAR_english_EMA_cleaned20220817.csv', encoding="utf-8-sig") as csvfile: 
    reader = csv.DictReader(csvfile)
    for row in reader:
        file_name = 'textfile\{0}.txt'.format(row['row_num'])
        with open(file_name, 'w') as f:
            f.write(row['transcription'])
```

### After running TAASSC, import back


```python
# Read in TAASSC analysis file
TAASSC = pd.read_csv(r'textfile\results_FINAL_sca.csv')
TAASSC['filename'] = TAASSC['filename'].replace('.txt','', regex=True)
TAASSC['filename'] = TAASSC['filename'].replace(r'C:\\Users\\yijun\\AppData\\Local\\Temp\\_.*?\\sca_parsed_files\\', '', regex=True)

# Keep only relevant syntactic complexity columns and rename them
TAASSC = TAASSC[['filename','MLC','CN_C']]
TAASSC = TAASSC.rename(columns={"filename": "row_num",'CN_C':'CNC'})

#convert row_num into int
TAASSC['row_num']=TAASSC['row_num'].astype(int)
TAASSC.sort_values(by='row_num',  inplace=True, ascending=True)
TAASSC.to_csv(rf'TAASSC\results_FINAL_sca.csv', index=False)

# Merge TAASSC with Clean File
df3=pd.merge(df2, TAASSC, on=["row_num"], how="left")

# replace MLC, CNC missing to 0
df3.fillna({'MLC': 0, 'CNC': 0}, inplace=True)
```

## Usage of Unique Words (Entropy)

### Shannon's Entropy


```python
import math

def entropy_calculator(data):
    if not data:
        return 0
    entropy = 0
    for x in range(256):
        p_x = float(data.count(chr(x)))/len(data)
        if p_x > 0:
            entropy += - p_x*math.log(p_x, 2)
    return entropy

df3['entropy']=df3['transcription'].apply(entropy_calculator)
```

### Applying ChaoShen Correction to Shannon's Entropy uses R package
### After running ChaoShen package in R, import as CSV


```python
# Read in ChaoShen analysis file
ChaoShen = pd.read_csv(r'data\EAR_english_20220820_ChaoShen.csv', encoding='cp1252')
ChaoShen = ChaoShen[['row_num','ChaoShen']]

# merge with df3
df4=pd.merge(df3, ChaoShen, on=["row_num"], how="left")

# replace ChaoShen missing to 0
df4.fillna({'ChaoShen': 0}, inplace=True)
```
