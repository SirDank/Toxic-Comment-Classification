import os
import sys
import json
import psutil
import subprocess
import numpy as np
import pandas as pd
import gradio as gr
import tensorflow as tf
from keras.optimizers import Adam
from matplotlib import pyplot as plt
from transformers import TFBertModel
from gradio.components import Textbox
from datetime import datetime, timedelta
from keras.utils import custom_object_scope
from keras.models import Sequential, Model, load_model
from concurrent.futures import ThreadPoolExecutor, as_completed
from keras.metrics import Precision, Recall, CategoricalAccuracy, BinaryAccuracy
from keras.layers import LSTM, Bidirectional, Embedding, TextVectorization, Conv1D, GlobalMaxPooling1D, MaxPooling1D, Input, Dropout, Dense, Flatten
from dankware import clr, cls, rm_line, magenta, white, red, get_duration, err, title, align

title("Toxic Comment Classification")
try: os.chdir(os.path.dirname(__file__))
except: pass
if not os.path.exists('plots'): os.mkdir('plots')
if not os.path.exists('json_files'): os.mkdir('json_files')
if not os.path.exists('trained_models'): os.mkdir('trained_models')
if not os.path.exists('json_files/training_time.json'): open("json_files/training_time.json","w+").write('{"LSTM": 0, "CNN": 0, "BERT": 0}')
for model_name in ['lstm', 'cnn', 'bert']:
    if not os.path.exists(f'json_files/{model_name}_history.json'):
        open(f'json_files/{model_name}_history.json', 'w+').write('{}')

def get_gpu_memory():
    command = "nvidia-smi --query-gpu=memory.free --format=csv"
    memory_free_info = subprocess.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
    memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
    return memory_free_values

def set_gpu_device():

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
    #physical_devices = tf.config.list_physical_devices('GPU')
    #if physical_devices:
        #print(clr("\n  > Found GPU"))
    try:
        tf.config.set_logical_device_configuration(tf.config.list_physical_devices('GPU')[0], [tf.config.LogicalDeviceConfiguration(memory_limit=get_gpu_memory()[0])])
        print(clr("\n  > Enabled GPU"))
        tf.keras.mixed_precision.set_global_policy('mixed_float16')
    except IndexError: pass
    except: print(clr(err(sys.exc_info()), 2))
    
set_gpu_device()

def load_data():

    df = pd.read_csv('train.csv')
    X = df['comment_text']
    y = df[df.columns[2:]].values
    vectorizer = TextVectorization(max_tokens=MAX_FEATURES, output_sequence_length=MAX_SEQUENCE_LENGTH, output_mode='int')
    vectorizer.adapt(X.values)
    vectorized_text = vectorizer(X.values)
    dataset = tf.data.Dataset.from_tensor_slices((vectorized_text, y))
    dataset = dataset.cache().shuffle(160000).batch(batch_size).prefetch(8)
    dataset_length = len(dataset)
    
    train = dataset.take(int(dataset_length * train_length))
    val = dataset.skip(int(dataset_length * train_length)).take(int(dataset_length * val_length))
    test = dataset.skip(int(dataset_length * (train_length + val_length))).take(int(dataset_length * test_length))

    return train, val, test, vectorizer, df

# single threaded

'''def create_embedding_layer():
    
    embedding_dims = 300
    max_features = MAX_FEATURES + 1
    
    # Load GloVe embeddings
    embeddings_index = {}
    with open('glove.840B.300d.txt', encoding='utf8') as f: # glove.6B.300d.txt
        for line in f:
            values = line.split(' ')
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    
    # Create embedding matrix
    embedding_matrix = np.zeros((max_features, embedding_dims))
    for index, word in enumerate(vectorizer.get_vocabulary()):
        if index > max_features - 1:
            break
        else:
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[index] = embedding_vector
                
    return embedding_matrix, embedding_dims'''

# multi threaded

def get_embedding_vector(embeddings_index, word, index):
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        return index, embedding_vector
    else:
        return index, np.zeros(300, dtype='float32')

def create_embedding_layer():
    
    print(clr(f"\n  > Creating Embedding Layer..."))
    embedding_dims = 300
    max_features = MAX_FEATURES + 1
    
    # Load GloVe embeddings
    embeddings_index = {}
    with open('optimised_glove.840B.300d.txt', 'r', encoding='utf8', errors='ignore') as f:
        for line in f:
            values = line.split(' ')
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    
    # Create embedding matrix
    embedding_matrix = np.zeros((max_features, embedding_dims))
    with ThreadPoolExecutor() as executor:
        futures = []
        for index, word in enumerate(vectorizer.get_vocabulary()):
            if index > max_features - 1:
                break
            else:
                futures.append(executor.submit(get_embedding_vector, embeddings_index, word, index))
                
        for future in as_completed(futures):
            index, embedding_vector = future.result()
            embedding_matrix[index] = embedding_vector
                
    return embedding_matrix, embedding_dims

# Define and compile models

def build_lstm_model():

    embedding_matrix, embedding_dims = create_embedding_layer()
    print(clr(f"\n  > Building {model_type} Model..."))
    model = Sequential()
    model.add(Embedding(input_dim=MAX_FEATURES + 1, output_dim=embedding_dims, embeddings_initializer=tf.keras.initializers.Constant(embedding_matrix), trainable=False, mask_zero=True,))
    model.add(Bidirectional(LSTM(32, activation='tanh')))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(DROPOUT_RATE))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(DROPOUT_RATE))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(DROPOUT_RATE))
    model.add(Dense(6, activation='sigmoid'))
    model.compile(loss='BinaryCrossentropy', optimizer=optimizer, metrics=metrics)
    return model

def build_cnn_model():

    FILTERS = 250
    KERNEL_SIZE = 3
    HIDDEN_DIMS = 250
    embedding_matrix, embedding_dims = create_embedding_layer()
    print(clr(f"\n  > Building {model_type} Model..."))
    model = Sequential()
    model.add(Embedding(input_dim=MAX_FEATURES + 1, output_dim=embedding_dims, embeddings_initializer=tf.keras.initializers.Constant(embedding_matrix), trainable=False, mask_zero=True,))
    model.add(Conv1D(FILTERS, KERNEL_SIZE, activation='relu'))
    model.add(MaxPooling1D())
    model.add(Dropout(DROPOUT_RATE))
    model.add(Conv1D(FILTERS, KERNEL_SIZE, activation='relu'))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(HIDDEN_DIMS, activation='relu'))
    model.add(Dropout(DROPOUT_RATE))
    model.add(Dense(6, activation='sigmoid'))
    model.compile(loss='BinaryCrossentropy', optimizer=optimizer, metrics=metrics)
    return model

def build_bert_model():
    print(clr(f"\n  > Building {model_type} Model..."))
    bert_layer = TFBertModel.from_pretrained('prajjwal1/bert-tiny', from_pt=True)
    input_layer = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype=tf.int32, name="input_layer")
    sequence_output = bert_layer(input_layer)[0]
    bert = Dropout(DROPOUT_RATE)(sequence_output)
    bert = Dense(256, activation='relu')(bert)
    bert = Dropout(DROPOUT_RATE)(bert)
    bert = Flatten()(bert)
    output_layer = Dense(6, activation='sigmoid')(bert)
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer=optimizer,loss='BinaryCrossentropy',metrics=metrics)
    return model

'''def model_checkpoint():
    global train_checkpoint_callback, val_checkpoint_callback, early_stop_callback_1, early_stop_callback_2
    train_checkpoint_callback = ModelCheckpoint(f'trained_models/{model_type.lower()}_train_checkpoint.h5', monitor='loss', mode='min', save_best_only=True, save_weights_only=True)
    val_checkpoint_callback = ModelCheckpoint(f'trained_models/{model_type.lower()}_val_checkpoint.h5', monitor='val_loss', mode='min', save_best_only=True, save_weights_only=True)
    early_stop_callback_1 = EarlyStopping(monitor='loss', mode='min', patience=3)
    early_stop_callback_2 = EarlyStopping(monitor='val_loss', mode='min', patience=3)'''

def train_model():

    global _history

    history_new = model.fit(train, epochs=EPOCHS, validation_data=val, use_multiprocessing=True, workers=-1) #, callbacks=[train_checkpoint_callback, val_checkpoint_callback, early_stop_callback_1, early_stop_callback_2])
    if _history != {}:
        for key in _history:
            _history[key].extend(history_new.history[key])
    else:
        _history = history_new.history
    open(f'json_files/{model_type.lower()}_history.json','w+').write(json.dumps(_history))
    
    loss = _history['val_loss'][-1]
    precision = _history['val_precision'][-1]
    recall = _history['val_recall'][-1]
    accuracy = _history['val_accuracy'][-1]
    
    return loss, precision, recall, accuracy

def save_timings():
    log = json.loads(open("json_files/training_time.json","r").read())
    log[model_type] = int(log[model_type] + (datetime.now() - training_time).seconds)
    to_print = f"\n  > Total Training Time: {get_duration(timedelta(0), timedelta(seconds = log[model_type]))}"
    open("log.txt", "a+").write(to_print.replace('  > ','\n'))
    open("json_files/training_time.json","w").write(json.dumps(log))
    print(clr(to_print))

# Scoring function for Gradio interface
def score_comment(comment):
    vectorized_comment = vectorizer([comment])
    print(f"\n{comment}")
    results = model.predict(vectorized_comment)
    print(results)
    text = ''
    for idx, col in enumerate(df.columns[2:]):
        text += '{}: {}\n'.format(col, results[0][idx] > 0.5)
    return text.strip()

# Set up Gradio interface
def run_interface():
    gradio_interface = gr.Interface(
        fn=score_comment,
        inputs=Textbox(placeholder='Enter your comment here...'),
        outputs=Textbox(),
        title='Toxic Comment Classification',
        description='Enter your comment and the model will predict the probability of it being toxic for each category.',
        #live=True,
    )

    return gradio_interface

def save_plot():
    plt.figure()
    pd.DataFrame(_history).plot()
    counter = 1
    while True:
        plot_path = f'plots/{model_type.lower()}_{counter}.png'
        if not os.path.exists(plot_path): break
        else: counter += 1
    plt.savefig(plot_path, dpi=1200)

if __name__ == '__main__':

    banner = '\n\n88888888888                d8b                .d8888b.                                      \n    888                    Y8P               d88P  Y88b                                     \n    888                                      888    888                                     \n    888   .d88b.  888  888 888  .d8888b      888         8888b.  888  888  .d88b.   .d88b.  \n    888  d88""88b `Y8bd8P\' 888 d88P"         888  88888     "88b 888  888 d88P"88b d8P  Y8b \n    888  888  888   X88K   888 888           888    888 .d888888 888  888 888  888 88888888 \n    888  Y88..88P .d8""8b. 888 Y88b.         Y88b  d88P 888  888 Y88b 888 Y88b 888 Y8b.     \n    888   "Y88P"  888  888 888  "Y8888P       "Y8888P88 "Y888888  "Y88888  "Y88888  "Y8888  \n                                                                               888          \n                                                                          Y8b d88P          \n                                                                           "Y88P"           \n'
    cls(); print(align(clr(banner, 4)))
    print(align(f"{white}made by {magenta}sir{white}.{magenta}dank {white}with {red}<3"))
    
    TRAINING_MODE = False
    SAVE_PLOT = False
    metrics = [Precision(name='precision'), Recall(name='recall'), BinaryAccuracy(name='accuracy')] # CategoricalAccuracy(name='accuracy')
    optimizer = Adam(learning_rate=3e-5) # AdamW(learning_rate=3e-5, epsilon=1e-08, weight_decay=0.01, clipnorm=1.0)
    
    #for model_type, repeats in zip(['CNN', 'LSTM', 'BERT'], [1,1,1]):
    #for _ in range(repeats):
    for _ in range(1):

        # Select model architecture
        print("")
        while True:
            if not TRAINING_MODE:
                model_type = input(clr("  > Enter Model Architecture [ LSTM / CNN / BERT ]: ") + magenta).upper()
            if model_type in ['LSTM', 'CNN', 'BERT']: break
            else: rm_line()

        DROPOUT_RATE = 0.2
        train_length = 0.1
        val_length = 0.1
        test_length = 0.1
        
        # NOTE: reduce batch size if running out of memory
        
        if model_type in ['LSTM', 'CNN']:
            EPOCHS = 10
            batch_size = 128 # 1024
            MAX_FEATURES = 220000
            MAX_SEQUENCE_LENGTH = 1800
            #train_length = 0.7
            #val_length = 0.2
            #test_length = 1 - train_length - val_length
        elif model_type == 'BERT':
            EPOCHS = 5
            batch_size = 64
            MAX_FEATURES = 30521
            MAX_SEQUENCE_LENGTH = 512
            #train_length = 0.1
            #val_length = 0.1
            #test_length = 0.1

        train, val, test, vectorizer, df = load_data()

        if os.path.exists(f"trained_models/{model_type}.h5"):
            print(clr(f"\n  > Loading {model_type} Model..."))
            with custom_object_scope({'TFBertModel': TFBertModel}):
                model = load_model(f"trained_models/{model_type}.h5")
        else:
            if model_type == "LSTM": model = build_lstm_model()
            elif model_type == "CNN": model = build_cnn_model()
            elif model_type == "BERT": model = build_bert_model()
            
        if TRAINING_MODE or not os.path.exists(f"trained_models/{model_type}.h5"):
            #print(clr(f"\n  > Saving Model Checkpoints..."))
            #model_checkpoint()
            print(clr(f"\n  > Training {model_type} Model...\n"))
            _history = json.loads(open(f'json_files/{model_type.lower()}_history.json','r').read())
            training_time = datetime.now()
            loss, precision, recall, accuracy = train_model()
            save_timings()
            print(clr(f"\n  > Saving {model_type} Model..."))
            
            save_path = f"trained_models/{model_type}.h5"
            backup_path = f"trained_models/{model_type}_backup.h5"
            
            if os.path.exists(save_path):
                if os.path.exists(backup_path):
                    os.remove(backup_path)
                os.rename(save_path, backup_path)

            model.save(save_path) #, save_best_only=True, save_weights_only=True, mode='min')

            to_print = '\n  > {} Model Loss: {:.2f}%'.format(model_type, loss * 100) + \
                '\n  > {} Model Precision: {:.2f}%'.format(model_type, precision * 100) + \
                '\n  > {} Model Recall: {:.2f}%'.format(model_type, recall * 100) + \
                '\n  > {} Model Accuracy: {:.2f}%'.format(model_type, accuracy * 100)
            print(clr(to_print))
            open("log.txt", "a+").write('\n' + to_print)
        
        if not TRAINING_MODE:
            print(clr("\n  > Starting Gradio Interface...\n"))
            gradio_interface = run_interface()
            gradio_interface.launch()

        if SAVE_PLOT:
            print(clr("\n  > Saving plot..."))
            save_plot()

