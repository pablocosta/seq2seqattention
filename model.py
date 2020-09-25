import tensorflow_addons as tfa
import tensorflow as tf

embedding_dims = 256
rnn_units = 1024
dense_units = 1024
BATCH_SIZE = 64



# ENCODER
class EncoderNetwork(tf.keras.Model):
    def __init__(self, input_vocab_size, embedding_dims, rnn_units):
        super().__init__()
        self.encoder_embedding = tf.keras.layers.Embedding(input_dim=input_vocab_size,
                                                           output_dim=embedding_dims)
        self.encoder_rnnlayer = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(rnn_units, return_sequences=True,
                                                     return_state=True))

# DECODER


class DecoderNetwork(tf.keras.Model):
    def __init__(self, output_vocab_size, embedding_dims, rnn_units, Tx, attnform="Luong"):
        super().__init__()
        self.decoder_embedding = tf.keras.layers.Embedding(input_dim=output_vocab_size,
                                                           output_dim=embedding_dims)
        self.dense_layer = tf.keras.layers.Dense(output_vocab_size)
        self.decoder_rnncell = tf.keras.layers.LSTMCell(rnn_units)
        # Sampler
        self.sampler = tfa.seq2seq.sampler.TrainingSampler()
        # Create attention mechanism with memory = None
        self.attention_mechanism = self.build_attention_mechanism(
            dense_units, None, BATCH_SIZE*[Tx], form=attnform)
        self.rnn_cell = self.build_rnn_cell(BATCH_SIZE)
        self.decoder = tfa.seq2seq.BasicDecoder(self.rnn_cell, sampler=self.sampler,
                                                output_layer=self.dense_layer)

    def build_attention_mechanism(self, units, memory, memory_sequence_length, form="Luong"):

        if form == "Luong":
            return tfa.seq2seq.LuongAttention(units, memory=memory, memory_sequence_length=memory_sequence_length)
        elif form == "Bahdanau":
            return tfa.seq2seq.BahdanauAttention(units, memory=memory, memory_sequence_length=memory_sequence_length)

    # wrap decodernn cell
    def build_rnn_cell(self, batch_size):
        rnn_cell = tfa.seq2seq.AttentionWrapper(self.decoder_rnncell, self.attention_mechanism,
                                                attention_layer_size=dense_units)
        return rnn_cell

    def build_decoder_initial_state(self, batch_size, encoder_state, Dtype):
        decoder_initial_state = self.rnn_cell.get_initial_state(batch_size=batch_size,
                                                                dtype=Dtype)
        decoder_initial_state = decoder_initial_state.clone(
            cell_state=encoder_state)
        return decoder_initial_state

def loss_function(y_pred, y):
   
    #shape of y [batch_size, ty]
    #shape of y_pred [batch_size, Ty, output_vocab_size] 
    sparsecategoricalcrossentropy = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True,
                                                                                  reduction='none')
    loss = sparsecategoricalcrossentropy(y_true=y, y_pred=y_pred)
    mask = tf.logical_not(tf.math.equal(y,0))   #output 0 for y=0 else output 1
    mask = tf.cast(mask, dtype=loss.dtype)
    loss = mask* loss
    loss = tf.reduce_mean(loss)
    return loss

#RNN LSTM hidden and memory state initializer
def initialize_initial_state():
        return [tf.zeros((BATCH_SIZE, rnn_units)), tf.zeros((BATCH_SIZE, rnn_units))]