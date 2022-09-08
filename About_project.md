# Introduction

Where else but Quora can a physicist help a chef with a math problem and get cooking tips in return? Quora is a place to gain and share knowledge—about anything. It’s a platform to ask questions and connect with people who contribute unique insights and quality answers. This empowers people to learn from each other and to better understand the world.

Over 100 million people visit Quora every month, so it's no surprise that many people ask similarly worded questions. Multiple questions with the same intent can cause seekers to spend more time finding the best answer to their question, and make writers feel they need to answer multiple versions of the same question. Quora values canonical questions because they provide a better experience to active seekers and writers, and offer more value to both of these groups in the long term.

By Trying different classification algorithms we have to find best optimised ml model which can best identify duplicate questions. We use natural language processing by applying advanced techniques to classify whether question pairs are duplicates or not. Doing so will make it easier to find high quality answers to questions resulting in an improved experience for Quora writers, seekers, and readers.

# Problem Statement

Identify which questions asked on Quora are duplicates of questions that have already been asked.

This could be useful to instantly provide answers to questions that have already been answered.

We are tasked with predicting whether a pair of questions are duplicates or not.

# Real world/Business Objectives and Constraints

The cost of a mis-classification can be very high.

You would want a probability of a pair of questions to be duplicates so that you can choose any threshold of choice.

No strict latency concerns.

Interpretability is partially important.

# Quora Data Matrix 

In our data we initially have 5 features ID of a data point , Question Id 1, Question Id 2, Question 1(text), Question 2(text) and is_duplicate(Binary)
we have total of 404290 data points in our dataset.

Approximately 63% of our data point are not similar question is_duplicate = 0 and 36% of the questions are similar ie is_duplicate = 1.

In our data all questions are not unique but all the pairs are unique the similar question can we tested with multiple different question.
total no of unique question are 537933

total no of question that appear more tha one time 111780

max no of time a single question is repeated 157 times

# Basic Feature Extraction ( before cleaning )

Let us now construct a few features like :

 • freq_qid1 = Frequency of qid1's
 
 • freq_qid2 = Frequency of qid2's
 
 • q1len = Length of q1
 
 • q2len = Length of q2
 
 • q1_n_words = Number of words in Question 1
  
 • q2_n_words = Number of words in Question 2
 
 • word_Common = ( Number of common unique words in Question 1 and Question 2 )
 
 • word_Total = ( Total num of words in Question 1 + Total num of words in Question 2 )
 
 • word_share = ( word_common ) / ( word_Total )
 
 • freq_q1 + freq_q2 = sum total of frequency of qid1 and qid2
 
 • freq_q1 - freq_q2 = absolute difference of frequency of qid1 and qid2
 
 # Preprocessing of Text
 
 • Preprocessing :
 
    
    ▪ Removing html tags
    
    ▪ Removing Punctuations
    
    ▪ Performing stemming
    
    ▪ Removing Stopwords
    
    ▪ Expanding contractions etc.
    
# Advanced Feature Extraction (NLP and Fuzzy Features)
Definition:

Token: You get a token by splitting sentence a space

Stop_Word : stop words as per NLTK.

Word : A token that is not a stop_word

Features:

cwc_min : Ratio of common_word_count to min lenghth of word count of Q1 and Q2
cwc_min = common_word_count / (min(len(q1_words), len(q2_words))



cwc_max : Ratio of common_word_count to max lenghth of word count of Q1 and Q2
cwc_max = common_word_count / (max(len(q1_words), len(q2_words))



csc_min : Ratio of common_stop_count to min lenghth of stop count of Q1 and Q2
csc_min = common_stop_count / (min(len(q1_stops), len(q2_stops))



csc_max : Ratio of common_stop_count to max lenghth of stop count of Q1 and Q2
csc_max = common_stop_count / (max(len(q1_stops), len(q2_stops))



ctc_min : Ratio of common_token_count to min lenghth of token count of Q1 and Q2
ctc_min = common_token_count / (min(len(q1_tokens), len(q2_tokens))



ctc_max : Ratio of common_token_count to max lenghth of token count of Q1 and Q2
ctc_max = common_token_count / (max(len(q1_tokens), len(q2_tokens))



last_word_eq : Check if First word of both questions is equal or not
last_word_eq = int(q1_tokens[-1] == q2_tokens[-1])



first_word_eq : Check if First word of both questions is equal or not
first_word_eq = int(q1_tokens[0] == q2_tokens[0])



abs_len_diff : Abs. length difference
abs_len_diff = abs(len(q1_tokens) - len(q2_tokens))



mean_len : Average Token Length of both Questions
mean_len = (len(q1_tokens) + len(q2_tokens))/2



fuzz_ratio : https://github.com/seatgeek/fuzzywuzzy#usage http://chairnerd.seatgeek.com/fuzzywuzzy-fuzzy-string-matching-in-python/



fuzz_partial_ratio : https://github.com/seatgeek/fuzzywuzzy#usage http://chairnerd.seatgeek.com/fuzzywuzzy-fuzzy-string-matching-in-python/



token_sort_ratio : https://github.com/seatgeek/fuzzywuzzy#usage http://chairnerd.seatgeek.com/fuzzywuzzy-fuzzy-string-matching-in-python/

token_set_ratio : https://github.com/seatgeek/fuzzywuzzy#usage http://chairnerd.seatgeek.com/fuzzywuzzy-fuzzy-string-matching-in-python/

longest_substr_ratio : Ratio of length longest common substring to min lenghth of token count of Q1 and Q2

longest_substr_ratio = len(longest common substring) / (min(len(q1_tokens), len(q2_tokens))  

# Performance measurement of models

Accourding to our data and type of problem we are solving we are using log loss and confusion metrix to analyse the performce of our models 

# ML models 

By our pre processed data and our problem statement we test different models and check their performance here we are using logistic regression, Random forest, svm and xg boost to classify the question .

XG boost is best performing model with minimum error
