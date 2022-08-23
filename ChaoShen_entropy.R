rm(list = ls())

library(ggplot2)
library(wesanderson)
library(scales)
library(tidyverse)
library(scales)
library(koRpus.lang.en)
library(entropy)
library(stringr)
library(dplyr)




mydata <- read.csv("C:\\Users\\yijun\\Documents\\Python Practices\\data\\EAR_english_EMA_cleaned_forR.csv", fileEncoding="UTF-8-BOM")
transcription <- mydata$transcription
mydata$cleantext <- transcription %>%
  str_to_lower() %>%
  str_replace_all("[:punct:]", "")


# Lemmatize0
Lemmatize = function(word, print =T, debug = F){
  if (print == T){
    print(word)}
  if(!is.na(word)){
    if(word == ""){
      return("")}}
  lemmax = koRpus::treetag(as.character(word), treetagger="manual", format="obj", debug = debug, TT.tknz=T, lang="en", TT.options=list(path="C:\\treetagger", preset="en"))
  if(lemmax@tokens[["lemma"]] == "<unknown>"){
    if (print == T){
      print(lemmax@tokens[["token"]])}
    return (lemmax@tokens[["token"]])}
  else{
    if (print == T){
      print(lemmax@tokens[["lemma"]])}
    return(lemmax@tokens[["lemma"]])
  }}


# Lemmatize
mydata$Lemmatized <- sapply(mydata$cleantext, Lemmatize)

# turn it into numeric
tb <- mydata$Lemmatized
tb1 <- sapply(tb, unlist)
tb1[] <- sapply(tb1, factor)
asnumeric <- sapply(tb1, as.numeric)

#count frequency
frequency <- sapply(asnumeric, table)

#unlist
frequency_unlist <- sapply(frequency, as.numeric)

#entropy
mydata$ChaoShen<- sapply(frequency_unlist, entropy, unit= "log2", method = "CS")

#save subset of dataframe
ChaoShen = subset(mydata, select = -c(transcription, cleantext, Lemmatized))
write.csv(ChaoShen,"C:\\Users\\yijun\\Documents\\Python Practices\\data\\EAR_english_20220820_ChaoShen.csv", row.names = FALSE)