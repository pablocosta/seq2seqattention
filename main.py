from utils import *
from model import *
import tensorflow as tf
from sklearn.model_selection import train_test_split

NUM_EXAMPLES = 10000 # Limit dataset size

# load from .txt
filename = 'deu.txt' #change filename if necessary
dataset = load_dataset(filename)
# get pairs limited into 1000
pairs = to_pairs(dataset, limit=NUM_EXAMPLES)
print(f"Dataset size: {len(pairs)}")
raw_data_en, raw_data_ge = dataset_preprocess(pairs)

# show last 5 pairs
for pair in zip(raw_data_en[-5:],raw_data_ge[-5:]):
    print(pair)
    
en_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
en_tokenizer.fit_on_texts(raw_data_en)

data_en = en_tokenizer.texts_to_sequences(raw_data_en)
data_en = tf.keras.preprocessing.sequence.pad_sequences(data_en,padding='post')

ge_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
ge_tokenizer.fit_on_texts(raw_data_ge)

data_ge = ge_tokenizer.texts_to_sequences(raw_data_ge)
data_ge = tf.keras.preprocessing.sequence.pad_sequences(data_ge,padding='post')

X_train,  X_test, Y_train, Y_test = train_test_split(data_en,data_ge,test_size=0.2)
BUFFER_SIZE = len(X_train)
Tx = max_len(data_en)
Ty = max_len(data_ge)  


input_vocab_size = len(en_tokenizer.word_index)+1  
output_vocab_size = len(ge_tokenizer.word_index)+ 1
dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train)).shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
example_X, example_Y = next(iter(dataset))
print(example_X.shape) 
print(example_Y.shape) 

encoderNetwork = EncoderNetwork(input_vocab_size,embedding_dims, rnn_units)
decoderNetwork = DecoderNetwork(output_vocab_size,embedding_dims, rnn_units, Tx)
optimizer = tf.keras.optimizers.Adam()

def train_step(input_batch, output_batch,encoder_initial_cell_state):
    
    #initialize loss = 0
    loss = 0
    with tf.GradientTape() as tape:
        encoder_emb_inp = encoderNetwork.encoder_embedding(input_batch)
        a, a_tx, c_tx = encoderNetwork.encoder_rnnlayer(encoder_emb_inp, 
                                                        initial_state =encoder_initial_cell_state)

        #[last step activations,last memory_state] of encoder passed as input to decoder Network
        
         
        # Prepare correct Decoder input & output sequence data
        decoder_input = output_batch[:,:-1] # ignore <end>
        #compare logits with timestepped +1 version of decoder_input
        decoder_output = output_batch[:,1:] #ignore <start>


        # Decoder Embeddings
        decoder_emb_inp = decoderNetwork.decoder_embedding(decoder_input)

        #Setting up decoder memory from encoder output and Zero State for AttentionWrapperState
        decoderNetwork.attention_mechanism.setup_memory(a)
        decoder_initial_state = decoderNetwork.build_decoder_initial_state(BATCH_SIZE,
                                                                           encoder_state=[a_tx, c_tx],
                                                                           Dtype=tf.float32)
        
        #BasicDecoderOutput        
        outputs, _, _ = decoderNetwork.decoder(decoder_emb_inp,initial_state=decoder_initial_state,
                                               sequence_length=BATCH_SIZE*[Ty-1])

        logits = outputs.rnn_output
        #Calculate loss

        loss = loss_function(logits, decoder_output)

    #Returns the list of all layer variables / weights.
    variables = encoderNetwork.trainable_variables + decoderNetwork.trainable_variables  
    # differentiate loss wrt variables
    gradients = tape.gradient(loss, variables)

    #grads_and_vars – List of(gradient, variable) pairs.
    grads_and_vars = zip(gradients,variables)
    optimizer.apply_gradients(grads_and_vars)
    return loss

steps_per_epoch = BUFFER_SIZE//BATCH_SIZE

epochs = 15
for i in range(1, epochs+1):

    encoder_initial_cell_state = initialize_initial_state()
    total_loss = 0.0

    for ( batch , (input_batch, output_batch)) in enumerate(dataset.take(steps_per_epoch)):
        batch_loss = train_step(input_batch, output_batch, encoder_initial_cell_state)
        total_loss += batch_loss
    
    print("total loss: {} epoch {} ".format(total_loss.numpy(), i))