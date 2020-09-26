#########################################################################################




from process_seplines import *
from sklearn.model_selection import train_test_split
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
from datetime import datetime
import os
import argparse
import sys
import bert
import os
import re
import nltk
import math
import operator
from bert import run_classifier
from bert import optimization
from bert import tokenization

# We'll set sequences to be at most 128 tokens long.
MAX_SEQ_LENGTH = 128
SUMMARY_LINE_COUNT = 5

data_path = "./NewDataset/train-act-nonact.csv"
test_path = "./NewDataset/Test-act-nonact.csv"
data_path1 = "./NewDataset/train-nonact-notuse.csv"
test_path1 = "./NewDataset/Test-nonact-notuse.csv"

BERT_path = "/home/devang/chandra/trans_meet/transformers/model_cards/google/bert_uncased_L-4_H-256_A-4"
OUTPUT_DIR = ""
BERT_MODEL_HUB = "https://tfhub.dev/google/small_bert/bert_uncased_L-4_H-256_A-4/1"

PERCENT_TO_RETAIN = 10 
# label_list is the list of labels, i.e. True, False or 0, 1 or 'dog', 'cat'
label_list = [1, 2]
labels = [1, 2]

def init_params():
    #Load the dataset
    train = pd.read_csv(data_path)
    labels = []

    #print (train.head())
    #print (train.Label.value_counts())

    test = pd.read_csv(test_path)
    #print (test.head())
    #print(len(label_list))

    return train, test

def init_bert_params(train, test):
    # This BERT package bert-classifier which requires tensorflow.contrib which is not there in current tensorflow version 2.0 so we need to select tensorflow 1.x which can be done by this command

    # BERT Implementation


    # Installing BERT tensor flow 1.0.1 to resolve the error of Trying to access flag --preserve_unused_tokens before flags were parsed.


    #get_ipython().system('pip install bert-tensorflow==1.0.1')
    # Chandra
    os.system('pip install bert-tensorflow==1.0.1')

    DATA_COLUMN = 'Text'
    LABEL_COLUMN = 'Label'
    print('***** Model output directory: {} *****'.format(OUTPUT_DIR))
    # BERT
    # Use the InputExample class from BERT's run_classifier code to create examples from the data
    train_InputExamples = train.apply(lambda x: bert.run_classifier.InputExample(guid=None, # Globally unique ID for bookkeeping, unused in this example
                                                                       text_a = x[DATA_COLUMN], 
                                                                       text_b = None, 
                                                                       label = x[LABEL_COLUMN]), axis = 1)

    test_InputExamples = test.apply(lambda x: bert.run_classifier.InputExample(guid=None, 
                                                                       text_a = x[DATA_COLUMN], 
                                                                       text_b = None, 
                                                                       label = x[LABEL_COLUMN]), axis = 1)


    # This is a path to an uncased (all lowercase) version of BERT
    #BERT_MODEL_HUB = "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1"
    #BERT_MODEL_HUB ="/content/drive/My Drive/DLCP/NLP/uncased_L-2_H-256_A-4.zip"
    tokenizer = create_tokenizer_from_hub_module()

    tokenizer.tokenize("This here's an example of using the BERT tokenizer")


    # Convert our train and test features to InputFeatures that BERT understands.
    train_features = bert.run_classifier.convert_examples_to_features(train_InputExamples, label_list, MAX_SEQ_LENGTH, tokenizer)
    test_features = bert.run_classifier.convert_examples_to_features(test_InputExamples, label_list, MAX_SEQ_LENGTH, tokenizer)

    return train_features, test_features, tokenizer

def create_tokenizer_from_hub_module():
  """Get the vocab file and casing info from the Hub module."""
  with tf.Graph().as_default():
    bert_module = hub.Module(BERT_MODEL_HUB)
    tokenization_info = bert_module(signature="tokenization_info", as_dict=True)
    with tf.Session() as sess:
      vocab_file, do_lower_case = sess.run([tokenization_info["vocab_file"],
                                            tokenization_info["do_lower_case"]])
      
  return bert.tokenization.FullTokenizer(
      vocab_file=vocab_file, do_lower_case=do_lower_case)

# Create BERT Model
def create_model(is_predicting, input_ids, input_mask, segment_ids, labels,
                 num_labels):
  """Creates a classification model."""

  bert_module = hub.Module(
      BERT_MODEL_HUB,
      trainable=True)
  bert_inputs = dict(
      input_ids=input_ids,
      input_mask=input_mask,
      segment_ids=segment_ids)
  bert_outputs = bert_module(
      inputs=bert_inputs,
      signature="tokens",
      as_dict=True)

  # Use "pooled_output" for classification tasks on an entire sentence.
  # Use "sequence_outputs" for token-level output.
  output_layer = bert_outputs["pooled_output"]

  hidden_size = output_layer.shape[-1].value

  # Create our own layer to tune for politeness data.
  output_weights = tf.get_variable(
      "output_weights", [num_labels, hidden_size],
      initializer=tf.truncated_normal_initializer(stddev=0.02))

  output_bias = tf.get_variable(
      "output_bias", [num_labels], initializer=tf.zeros_initializer())

  with tf.variable_scope("loss"):

    # Dropout helps prevent overfitting
    output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

    logits = tf.matmul(output_layer, output_weights, transpose_b=True)
    logits = tf.nn.bias_add(logits, output_bias)
    log_probs = tf.nn.log_softmax(logits, axis=-1)

    # Convert labels into one-hot encoding
    one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)

    predicted_labels = tf.squeeze(tf.argmax(log_probs, axis=-1, output_type=tf.int32))
    # If we're predicting, we want predicted labels and the probabiltiies.
    if is_predicting:
      return (predicted_labels, log_probs)

    # If we're train/eval, compute loss between predicted and actual label
    per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
    loss = tf.reduce_mean(per_example_loss)
    return (loss, predicted_labels, log_probs)

# model_fn_builder actually creates our model function
# using the passed parameters for num_labels, learning_rate, etc.
def model_fn_builder(num_labels, learning_rate, num_train_steps,
                     num_warmup_steps):
  """Returns `model_fn` closure for TPUEstimator."""
  def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
    """The `model_fn` for TPUEstimator."""

    input_ids = features["input_ids"]
    input_mask = features["input_mask"]
    segment_ids = features["segment_ids"]
    label_ids = features["label_ids"]

    is_predicting = (mode == tf.estimator.ModeKeys.PREDICT)
    
    # TRAIN and EVAL
    if not is_predicting:

      (loss, predicted_labels, log_probs) = create_model(
        is_predicting, input_ids, input_mask, segment_ids, label_ids, num_labels)

      train_op = bert.optimization.create_optimizer(
          loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu=False)

      # Calculate evaluation metrics. 
      def metric_fn(label_ids, predicted_labels):
        accuracy = tf.metrics.accuracy(label_ids, predicted_labels)
        f1_score = tf.contrib.metrics.f1_score(
            label_ids,
            predicted_labels)
        auc = tf.metrics.auc(
            label_ids,
            predicted_labels)
        recall = tf.metrics.recall(
            label_ids,
            predicted_labels)
        precision = tf.metrics.precision(
            label_ids,
            predicted_labels) 
        true_pos = tf.metrics.true_positives(
            label_ids,
            predicted_labels)
        true_neg = tf.metrics.true_negatives(
            label_ids,
            predicted_labels)   
        false_pos = tf.metrics.false_positives(
            label_ids,
            predicted_labels)  
        false_neg = tf.metrics.false_negatives(
            label_ids,
            predicted_labels)
        return {
            "eval_accuracy": accuracy,
            "f1_score": f1_score,
            "auc": auc,
            "precision": precision,
            "recall": recall,
            "true_positives": true_pos,
            "true_negatives": true_neg,
            "false_positives": false_pos,
            "false_negatives": false_neg
        }

      eval_metrics = metric_fn(label_ids, predicted_labels)

      if mode == tf.estimator.ModeKeys.TRAIN:
        return tf.estimator.EstimatorSpec(mode=mode,
          loss=loss,
          train_op=train_op)
      else:
          return tf.estimator.EstimatorSpec(mode=mode,
            loss=loss,
            eval_metric_ops=eval_metrics)
    else:
      (predicted_labels, log_probs) = create_model(
        is_predicting, input_ids, input_mask, segment_ids, label_ids, num_labels)

      predictions = {
          'probabilities': log_probs,
          'labels': predicted_labels
      }
      return tf.estimator.EstimatorSpec(mode, predictions=predictions)

  # Return the actual model function in the closure
  return model_fn


def train_model(train_features, test_features):
    # Compute train and warmup steps from batch size
    # These hyperparameters are copied from this colab notebook (https://colab.sandbox.google.com/github/tensorflow/tpu/blob/master/tools/colab/bert_finetuning_with_cloud_tpus.ipynb)
    #BATCH_SIZE = 32
    # Chandra
    BATCH_SIZE = 8
    LEARNING_RATE = 2e-5
    NUM_TRAIN_EPOCHS = 3.0
    # Warmup is a period of time where hte learning rate 
    # is small and gradually increases--usually helps training.
    WARMUP_PROPORTION = 0.1
    # Model configs
    SAVE_CHECKPOINTS_STEPS = 500
    SAVE_SUMMARY_STEPS = 100

    # Compute # train and warmup steps from batch size
    num_train_steps = int(len(train_features) / BATCH_SIZE * NUM_TRAIN_EPOCHS)
    num_warmup_steps = int(num_train_steps * WARMUP_PROPORTION)

    print(num_train_steps)

    # Specify outpit directory and number of checkpoint steps to save
    run_config = tf.estimator.RunConfig(
        model_dir=OUTPUT_DIR,
        save_summary_steps=SAVE_SUMMARY_STEPS,
        save_checkpoints_steps=SAVE_CHECKPOINTS_STEPS)

    model_fn = model_fn_builder(
      num_labels=len(label_list),
      learning_rate=LEARNING_RATE,
      num_train_steps=num_train_steps,
      num_warmup_steps=num_warmup_steps)

    estimator = tf.estimator.Estimator(
      model_fn=model_fn,
      config=run_config,
      params={"batch_size": BATCH_SIZE})

    # Create an input function for training. drop_remainder = True for using TPUs.
    train_input_fn = bert.run_classifier.input_fn_builder(
        features=train_features,
        seq_length=MAX_SEQ_LENGTH,
        is_training=True,
        drop_remainder=False)

    print(f'Beginning Training!')
    current_time = datetime.now()
    estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)
    print("Training took time ", datetime.now() - current_time)

    return estimator

def getPrediction(in_sentences, estimator, tokenizer):
  labels = [1,2]
  input_examples = [run_classifier.InputExample(guid="", text_a = x, text_b = None, label = 1) for x in in_sentences] # here, "" is just a dummy label
  input_features = run_classifier.convert_examples_to_features(input_examples, label_list, MAX_SEQ_LENGTH, tokenizer)
  predict_input_fn = run_classifier.input_fn_builder(features=input_features, seq_length=MAX_SEQ_LENGTH, is_training=False, drop_remainder=False)
  predictions = estimator.predict(predict_input_fn)
  #return [( labels[prediction['labels']]) for sentence, prediction in zip(in_sentences, predictions)]
  return [(sentence, prediction['probabilities'], labels[prediction['labels']]) for sentence, prediction in zip(in_sentences, predictions)]

def classify_sentences(estimator, tokenizer, test_features, testfile):
    test_input_fn = run_classifier.input_fn_builder(
        features=test_features,
        seq_length=MAX_SEQ_LENGTH,
        is_training=False,
        drop_remainder=False)

    estimator.evaluate(input_fn=test_input_fn, steps=None)

    #ul_test_path = os.path.join(project_path,'unlabeled_test_with_noise.tsv')
    #print(ul_test_path)
    ul_test = pd.read_csv(testfile, sep='\n')

    ul_test.head()

    ul_test.columns = ['Text']
    ul_test.head
    [count, col] = ul_test.shape
    print("count, col", count, col)
    dft2 = ul_test["Text"]

    pred_sentences = []
    for i in range (0,count):
      pred_sentences.append(dft2[i])

    print(len(pred_sentences))

    input_examples = [run_classifier.InputExample(guid="", text_a = x, text_b = None, label = 1) for x in pred_sentences]

    input_features = run_classifier.convert_examples_to_features(input_examples, label_list, MAX_SEQ_LENGTH, tokenizer)

    #pred_sentences = [
    #  "That movie was absolutely awful",
    #  "Please finish this work by Monday",
    #  "I am going to attend the meeting in Germany",
    #  "The results are very good this year. We can close the deal",
    #  "abhi said I oft manerism",
    #  " utkash said he also so anyone can be confusing.",
    #  " if the say otherwise in great in the dark but you know also I'll be able to text thing ."
    #]
    print(pred_sentences)

    predictions = getPrediction(pred_sentences, estimator, tokenizer)

    print (predictions)

    return predictions

def init_summary_params(testfile):
    # Text scoring with TF-IDF
    from nltk.corpus import stopwords
    from nltk.corpus import wordnet
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')
    from nltk.stem import WordNetLemmatizer
    from nltk.tokenize import sent_tokenize,word_tokenize
    from nltk.corpus import stopwords
    Stopwords = set(stopwords.words('english'))
    wordlemmatizer = WordNetLemmatizer()

    fname = testfile 
    fname = open(fname , 'r')
    text = fname.readlines()

    #text = text.strip('\n')
    newtext = []
    for index in range(len(text)):
        #print (text[index])
        line = text[index].strip()
        #if line[-1] != '.':
            #line = line+'.'
        newtext.append(line)

    print (newtext)
    # Chandra
    #tokenized_sentence = sent_tokenize(newtext)
    tokenized_sentence = newtext
    #tokenized_sentence = tokenized_sentence.split('\n')
    text = remove_special_characters(str(text))
    text = re.sub(r'\d+', '', text)
    tokenized_words_with_stopwords = word_tokenize(text)
    tokenized_words = [word for word in tokenized_words_with_stopwords if word not in Stopwords]
    tokenized_words = [word for word in tokenized_words if len(word) > 1]
    tokenized_words = [word.lower() for word in tokenized_words]
    tokenized_words = lemmatize_words(wordlemmatizer, tokenized_words)

    print(tokenized_sentence)

    no_of_sentences = int((PERCENT_TO_RETAIN * len(tokenized_sentence))/100)
    print("percentage of sentences count: ", no_of_sentences)
    print("length of tokenized sentence: ", len(tokenized_sentence))

    for sent in tokenized_sentence:
      print("---")
      print(sent)

    return tokenized_sentence, no_of_sentences, Stopwords, wordlemmatizer

# Preprocess 
def lemmatize_words(wordlemmatizer, words):
    lemmatized_words = []
    for word in words:
       lemmatized_words.append(wordlemmatizer.lemmatize(word))
    return lemmatized_words

def stem_words(words):
    stemmed_words = []
    for word in words:
       stemmed_words.append(stemmer.stem(word))
    return stemmed_words


# Remove special charectors
def remove_special_characters(text):
    regex = r'[^a-zA-Z0-9\s]'
    text = re.sub(regex,'',text)
    return text


# Calculate frequency of each word in document
def freq(words):
    words = [word.lower() for word in words]
    dict_freq = {}
    words_unique = []
    for word in words:
       if word not in words_unique:
          words_unique.append(word)
    for word in words_unique:
          dict_freq[word] = words.count(word)
    return dict_freq

# Calculate sentence score

# POS Tagging
def pos_tagging(text):
    pos_tag = nltk.pos_tag(text.split())
    pos_tagged_noun_verb = []
    for word,tag in pos_tag:
        if tag == "NN" or tag == "NNP" or tag == "NNS" or tag == "VB" or tag == "VBD" or tag == "VBG" or tag == "VBN" or tag == "VBP" or tag == "VBZ":
            pos_tagged_noun_verb.append(word)
    return pos_tagged_noun_verb

# Tf score
def tf_score(word,sentence):
    freq_sum = 0
    word_frequency_in_sentence = 0
    len_sentence = len(sentence)
    for word_in_sentence in sentence.split():
        if word == word_in_sentence:
            word_frequency_in_sentence = word_frequency_in_sentence + 1
    tf =  word_frequency_in_sentence/ len_sentence
    return tf

def idf_score(no_of_sentences,word,sentences, Stopwords, wordlemmatizer):
    no_of_sentence_containing_word = 0
    for sentence in sentences:
        sentence = remove_special_characters(str(sentence))
        sentence = re.sub(r'\d+', '', sentence)
        sentence = sentence.split()
        sentence = [word for word in sentence if word.lower() not in Stopwords and len(word)>1]
        sentence = [word.lower() for word in sentence]
        sentence = [wordlemmatizer.lemmatize(word) for word in sentence]
        if word in sentence:
            no_of_sentence_containing_word = no_of_sentence_containing_word + 1
    idf = math.log10(no_of_sentences/no_of_sentence_containing_word)
    return idf


# tf-idf score
def tf_idf_score(tf,idf):
   return tf*idf

# Word Tfidf
def word_tfidf(word,sentences,sentence, Stopwords, wordlemmatizer):
    word_tfidf = [] 
    tf = tf_score(word,sentence)
    idf = idf_score(len(sentences),word,sentences, Stopwords, wordlemmatizer)
    tf_idf = tf_idf_score(tf,idf)
    return tf_idf

def sentence_importance(sentence, sentences, Stopwords, wordlemmatizer):
     sentence_score = 0
     sentence = remove_special_characters(str(sentence))
     sentence = re.sub(r'\d+', '', sentence)
     pos_tagged_sentence = []
     #no_of_sentences = len(sentences)
     pos_tagged_sentence = pos_tagging(sentence)
     for word in pos_tagged_sentence:
         if word.lower() not in Stopwords and word not in Stopwords    and len(word)>1:
             word = word.lower() 
             word = wordlemmatizer.lemmatize(word)
             sentence_score = sentence_score + word_tfidf(word,sentences,sentence, Stopwords, wordlemmatizer)
     return sentence_score


def get_ranking_sentences(tokenized_sentence, no_of_sentences, Stopwords, wordlemmatizer):
    # Find most important setences
    c = 1
    sentence_with_importance = {}
    for sent in tokenized_sentence:
        sentenceimp = sentence_importance(sent,tokenized_sentence, Stopwords, wordlemmatizer)
        sentence_with_importance[c] = sentenceimp
        c = c+1
    sentence_with_importance = sorted(sentence_with_importance.items(), key=operator.itemgetter(1),reverse=True)
    cnt = 0
    summary = []
    sentence_no = []
    print("number of sentences: ", no_of_sentences)
    for word_prob in sentence_with_importance:
        if cnt < no_of_sentences:
            sentence_no.append(word_prob[0])
            cnt = cnt+1
        else:
          break
    sentence_no.sort()
    print("sentence number: ", sentence_no)
    cnt = 1
    for sentence in tokenized_sentence:
        if cnt in sentence_no:
           summary.append(sentence)
        cnt = cnt+1
    #summary = " ".join(summary)
    print("\n")
    print("Summary:")
    print(summary)

    outF = open(OUTPUT_DIR+'/summary.txt',"w")
    cnt = 0
    for sentence in summary:
        cnt = cnt+1
        if cnt == SUMMARY_LINE_COUNT:
            cnt = 0
            outF.write(sentence+'\n')
        else:
            outF.write(sentence+' ')

    outF.close()




if __name__ == "__main__":
    args=sys.argv[1:]
    parser = argparse.ArgumentParser(description="Parses command.")
    parser.add_argument("-f", "--transcript_filename", required=True, type=str, default=None, help="The name of the transcript file.")
    parser.add_argument("-o", "--output_dir", required=True, type=str, default=None, help="Output directory where actions, mom and others files to store.")
    parser.add_argument("-a", "--actions", required=False, default=None, help="Get only actions statements")
    parser.add_argument("-m", "--summary", required=False, default=None, help="Get only summary statements")
    parser.add_argument("-n", "--others", required=False, default=None, help="Get statements other than actions and summary statements")

    args = parser.parse_args(sys.argv[1:])

    if len(sys.argv) < 2:
        print('You have not specified the required arguments')
        print ("Use -h or --help for details of arguments")
        sys.exit()

    testfile = args.transcript_filename

    testfile = gen_sep_lines(args.transcript_filename)
    testfile = "./"+testfile

    OUTPUT_DIR = args.output_dir

    if args.actions:
        train, test = init_params()

        train_features, test_features, tokenizer = init_bert_params(train, test)

        estimator = train_model(train_features, test_features)

        predictions = classify_sentences(estimator, tokenizer, test_features, testfile)

        actions_fname = open(OUTPUT_DIR+"actions.txt", 'w')
        nonactions_fname = open(OUTPUT_DIR+"nonactions.txt", 'w')
        for ele in predictions:
            #return [(sentence, prediction['probabilities'], labels[prediction['labels']]) for sentence, prediction in zip(in_sentences, predictions)]
            if ele[2] == 2:
                actions_fname.write(ele[0]+'\n')
            else:
                nonactions_fname.write(ele[0]+'\n')
            
    
    if args.summary:

        tokenized_sentence, no_of_sentences, Stopwords, wordlemmatizer = init_summary_params(testfile)

        get_ranking_sentences(tokenized_sentence, no_of_sentences, Stopwords, wordlemmatizer)

    if args.actions == None and args.summary == None:
        train, test = init_params()

        train_features, test_features, tokenizer = init_bert_params(train, test)

        estimator = train_model(train_features, test_features)

        predictions = classify_sentences(estimator, tokenizer, test_features, testfile)
    
        tokenized_sentence, no_of_sentences, Stopwords, wordlemmatizer = init_summary_params(testfile)

        get_ranking_sentences(tokenized_sentence, no_of_sentences, Stopwords, wordlemmatizer)
