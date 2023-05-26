from transformers import BertTokenizer 
from transformers import TFBertModel 
from transformers import RobertaTokenizer 
from transformers import TFRobertaModel 
import tensorflow as tf 
from tensorflow import keras 
from tensorflow.keras.callbacks import ModelCheckpoint

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report 
from sklearn.metrics import multilabel_confusion_matrix
import numpy as np
import pandas as pd

from text_clean_utils import strip_all_entities, clean_hashtags, filter_chars, remove_mult_spaces
import argparse 
from pathlib import Path 
import os 

parser = argparse.ArgumentParser() 
parser.add_argument("-p","--path_to_dataset",default="./datasets/processed_df.csv", help="path to processed csv file ready to feed into the model. One hot encoding already done")
parser.add_argument("-m","--model",default='bert',choices=['bert','roberta']) 
parser.add_argument("--pre_trained", default = None, help = "path of pre-trained model to continue training")

args = parser.parse_args() 
args = vars(args)


class Model(): 
    def __init__(self, model_name = "bert", dataset_path = None, pre_trained = None) -> None:
        self.pre_trained = pre_trained
        self.name = model_name
        if model_name == "bert": 
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')  
            self.model = TFBertModel.from_pretrained('bert-base-uncased')
        elif model_name == "roberta": 
            self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base') 
            self.model = TFRobertaModel.from_pretrained('roberta-base')


        if not dataset_path: 
            self.X_train, self.Y_train = None, None  
            self.X_test, self.Y_test = None, None 
            self.X_val,  self.Y_val = None, None  
            self.MAX_LEN = None 

            df = pd.read_csv(dataset_path)
            self.feed_data(df)

            self.train_input_ids, self.train_attention_masks = self.tokenize(self.X_train)
            self.val_input_ids, self.val_attention_masks = self.tokenize(self.X_val)
            self.test_input_ids, self.test_attention_masks = self.tokenize(self.X_test)

            self.Y_train = self.Y_train.to_numpy() 
            self.Y_test = self.Y_test.to_numpy() 
            self.Y_val = self.Y_val.to_numpy()

        self.seq_model = self.create_model() 

    def create_model(self):
        opt = tf.keras.optimizers.Adam(learning_rate=1e-5)
        # loss = tf.keras.losses.CategoricalCrossentropy()
        loss = tf.keras.losses.BinaryCrossentropy()
        accuracy = tf.keras.metrics.CategoricalAccuracy() 
        precision = tf.keras.metrics.Precision() 
        recall = tf.keras.metrics.Recall()

        input_ids = tf.keras.Input(shape=(self.MAX_LEN,),dtype='int32')
        attention_masks = tf.keras.Input(shape=(self.MAX_LEN,),dtype='int32')
        embeddings = self.model([input_ids,attention_masks])[1]
        
        output = tf.keras.layers.Dense(256, activation="relu")(embeddings) 
        output = tf.keras.layers.Dense(64, activation="relu")(output) 
        # output = tf.keras.layers.Dense(7, activation ="softmax")(output)  
        output = tf.keras.layers.Dense(7, activation = "sigmoid")(output)
        # Sigmoid is better for our case because we need probability of every class independently, since an instance can belong to multiple classes as well

        seq_model = tf.keras.models.Model(inputs = [input_ids, attention_masks], outputs = output) 
        seq_model.compile(opt, loss=loss, metrics = [accuracy, precision, recall])
        if self.pre_trained is not None: 
            if os.path.exists(self.pre_trained): 
                try: 
                    seq_model.load_weights(self.pre_trained) 
                    print("loaded pre-trained model weights") 
                except: 
                    print("Unable to load pretrained model.")
            else: 
                print("path to pre-trained model not found") 
        else: 
            print("No Pre-trained model provided. Creating a fresh model with randomly initialized weights") 

        print(seq_model.summary())
        return seq_model 


    def feed_data(self, data): 
        # determine self.MAX_LEN 
        token_lens = []

        for txt in data['comment_text'].values:
            tokens = self.tokenizer.encode(txt, max_length=512, truncation=True)
            token_lens.append(len(tokens))
            
        max_len = np.max(token_lens)
        self.MAX_LEN = np.ceil(np.log2(max_len))
        print(f"MAX TOKENIZED SENTENCE LENGTH: {self.MAX_LEN}")


        # split data into train, test, val 
        X, Y = data.comment_text, data[['bias', 'religion', 'race', 'gender','political', 'lgbtq', 'none']]
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2,stratify = Y.bias)
        X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.2, stratify=Y_train.bias)  

        # clean text input
        X_train = X_train.map(strip_all_entities)
        X_train = X_train.map(clean_hashtags)
        X_train = X_train.map(filter_chars) 
        X_train = X_train.map(remove_mult_spaces)  

        X_test = X_test.map(strip_all_entities)
        X_test = X_test.map(clean_hashtags)
        X_test = X_test.map(filter_chars) 
        X_test = X_test.map(remove_mult_spaces)  

        X_val = X_val.map(strip_all_entities)
        X_val = X_val.map(clean_hashtags)
        X_val = X_val.map(filter_chars) 
        X_val = X_val.map(remove_mult_spaces)  

        self.X_train, self.Y_train, self.X_test, self.Y_test, self.X_val, self.Y_val = X_train, Y_train, X_test, Y_test, X_val, Y_val   


    def tokenize(self, data): 
        input_ids = []
        attention_masks = []
        ids = data.index 
        for i in ids:
            encoded = self.tokenizer.encode_plus(
                data[i],
                add_special_tokens=True,
                max_length=self.MAX_LEN,
                padding='max_length',
                return_attention_mask=True
            )
            input_ids.append(encoded['input_ids'])
            attention_masks.append(encoded['attention_mask'])
        return np.array(input_ids),np.array(attention_masks)
    

    def train(self, num_epochs=10, batch_size=32): 
        checkpoint_path = f'./trained_models/{self.name}_checkpoint.ckpt' 

        model_checkpoint_callback = ModelCheckpoint(
            filepath=checkpoint_path,
            save_weights_only=True,
            monitor='val_loss',
            mode='max',
            save_frequency="epoch")
        
        history_bert = self.seq_model.fit([self.train_input_ids,self.train_attention_masks], self.Y_train, validation_data=([self.val_input_ids,self.           val_attention_masks], self.Y_valid), epochs=num_epochs, batch_size=batch_size, callbacks = [model_checkpoint_callback])
        self.seq_model.save_weights(f'./trained_models/{self.name}_trained_{num_epochs}epochs.ckpt')


    def evaluate(self): 
        result = self.seq_model.predict([self.test_input_ids, self.test_attention_masks]) 
        y_pred = (result > 0.5).astype('int') 
        report  = classification_report(self.Y_test, y_pred) 
        print(report)