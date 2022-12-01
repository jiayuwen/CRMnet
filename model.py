import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Layer, Input, concatenate, Conv1D, Conv1DTranspose, \
    BatchNormalization, Dropout, Dense, Flatten, GlobalAveragePooling1D, MultiHeadAttention, LayerNormalization
from tensorflow.keras import Input


def return_model(model_name):
    model_dic={'crm_net': CRMnet()}
    return model_dic[model_name]

class SqueezeExcitation1DLayer(tf.keras.Model):

    def __init__(self, out_dim, ratio, layer_name="se"):
        super(SqueezeExcitation1DLayer, self).__init__(name=layer_name)
        self.squeeze = GlobalAveragePooling1D()
        self.excitation_a = Dense(units=out_dim / ratio, activation='relu')
        self.excitation_b = Dense(units=out_dim, activation='sigmoid')
        self.shape = [-1, 1, out_dim]

    def call(self, input_x):
        squeeze = self.squeeze(input_x)

        excitation = self.excitation_a(squeeze)
        excitation = self.excitation_b(excitation)

        scale = tf.reshape(excitation, self.shape)
        se = input_x * scale

        return se

## transformer code adopted from https://keras.io/examples/nlp/text_classification_with_transformer/
class TransformerBlock(Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = Sequential(
            [Dense(ff_dim, activation="relu"), Dense(embed_dim),]
        )
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output) # residual connection
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output) # residual connection

class PositionEmbedding(Layer):
    def __init__(self, max_len):
        super(PositionEmbedding, self).__init__()
        self.max_len = max_len

    def call(self, x):
        positions = tf.range(start=0, limit=111, delta=111/self.max_len)
        positions = tf.math.sin(positions)


        return tf.math.multiply(x, tf.expand_dims(tf.cast(positions, dtype='float32'), axis=-1))
    
def EncoderSeBlock(inputs, n_filters=15, kernel_size=7, dropout_prob=0.3, ratio=2, layer_name="",condensing=True):
    conv = Conv1D(n_filters,
                  kernel_size,  
                  activation='relu',
                  padding='same',
                  kernel_initializer='HeNormal')(inputs)
    se = SqueezeExcitation1DLayer(out_dim=n_filters, ratio=ratio, layer_name=layer_name+"_0")(conv)
    bn = BatchNormalization()(se)
    conv = Conv1D(n_filters,
                  kernel_size,  
                  activation='relu',
                  padding='same',
                  kernel_initializer='HeNormal')(bn)
    se = SqueezeExcitation1DLayer(out_dim=n_filters, ratio=ratio, layer_name=layer_name+"_1")(conv)
    conv = BatchNormalization()(se)

    if dropout_prob > 0:
        conv = tf.keras.layers.Dropout(dropout_prob)(conv)

    if condensing:
        next_layer = Conv1D(n_filters,kernel_size, activation='relu',strides=2, padding='same',kernel_initializer='HeNormal')(conv)
    else:
        next_layer = conv

    skip_connection = conv

    return next_layer, skip_connection

def DecoderSeBlock(prev_layer_input, skip_layer_input, kernel_size=7, ratio=2, layer_name="", n_filters=32):
    up = Conv1DTranspose(filters=n_filters,
                         kernel_size=kernel_size,
                         strides=2,
                         padding='same')(prev_layer_input)

    merge = concatenate([up, skip_layer_input], axis=-1)

    conv = Conv1D(n_filters,
                  kernel_size,
                  activation='relu',
                  padding='same',
                  kernel_initializer='HeNormal')(merge)
    se = SqueezeExcitation1DLayer(out_dim=n_filters, ratio=ratio, layer_name=layer_name+"_0")(conv)
    bn = BatchNormalization()(se)
    conv = Conv1D(n_filters,
                  kernel_size,  
                  activation='relu',
                  padding='same',
                  kernel_initializer='HeNormal')(bn)
    se = SqueezeExcitation1DLayer(out_dim=n_filters, ratio=ratio, layer_name=layer_name+"_1")(conv)
    bn = BatchNormalization()(se)
    return bn


def CRMnet(input_size=(112, 5), n_filters=128, n_head=12, kernel_size=11, n_ff_dims=128, n_classes=1):
    inputs = Input(input_size)
    max_len, vocab_size = input_size
    max_len = 14

    #Encoder
    cblock1 = EncoderSeBlock(inputs, n_filters, kernel_size=kernel_size, dropout_prob=0.1, layer_name="ecb_1_", condensing=True)
    cblock2 = EncoderSeBlock(cblock1[0], 128, kernel_size=kernel_size, dropout_prob=0.1, layer_name="ecb_2_", condensing=True)
    cblock3 = EncoderSeBlock(cblock2[0], 256, kernel_size=kernel_size, dropout_prob=0.1, layer_name="ecb_3_", condensing=True)
    cblock4 = EncoderSeBlock(cblock3[0], 256, kernel_size=kernel_size, dropout_prob=0.1, layer_name="ecb_4_", condensing=False)

    #Position Embedding
    embedding_layer_1 = PositionEmbedding(max_len)

    #Transformer Encoder
    transformer_block_1 = TransformerBlock(embed_dim=256, num_heads=n_head, ff_dim=n_ff_dims)
    transformer_block_2 = TransformerBlock(embed_dim=256, num_heads=n_head, ff_dim=n_ff_dims)
    transformer_block_3 = TransformerBlock(embed_dim=256, num_heads=n_head, ff_dim=n_ff_dims)
    transformer_block_4 = TransformerBlock(256, n_head, n_ff_dims)
    transformer_block_5 = TransformerBlock(256, n_head, n_ff_dims)
    transformer_block_6 = TransformerBlock(256, n_head, n_ff_dims)
    transformer_block_7 = TransformerBlock(256, n_head, n_ff_dims)
    transformer_block_8 = TransformerBlock(256, n_head, n_ff_dims)
    x = embedding_layer_1(cblock4[0])
    x = transformer_block_1(x)
    x1 = transformer_block_2(x)
    x = transformer_block_3(x) + x1
    x2 = transformer_block_4(x)
    x = transformer_block_5(x) + x2
    x3 = transformer_block_6(x) 
    x = transformer_block_7(x) + x3
    x = transformer_block_8(x)

    #Decoder
    ublock6 = DecoderSeBlock(x, cblock3[1], kernel_size=kernel_size, n_filters=128, layer_name="dcb_1_")
    ublock7 = DecoderSeBlock(ublock6, cblock2[1], kernel_size=kernel_size, n_filters=128, layer_name="dcb_2_")
    ublock8 = DecoderSeBlock(ublock7, cblock1[1], kernel_size=kernel_size, n_filters=128, layer_name="dcb_3_")

    # SE Block
    conv9_0 = Conv1D(filters=128,kernel_size=kernel_size,activation='relu',padding='same',kernel_initializer='he_normal')(ublock8)
    x = SqueezeExcitation1DLayer(out_dim=128, ratio=2, layer_name='se_0')(conv9_0)
    conv9_1 = Conv1D(filters=64,kernel_size=kernel_size,activation='relu',padding='same',kernel_initializer='he_normal')(x)
    x = SqueezeExcitation1DLayer(out_dim=64, ratio=2, layer_name='se_1')(conv9_1)
    conv9_2 = Conv1D(filters=32,kernel_size=kernel_size,activation='relu',padding='same',kernel_initializer='he_normal')(x)
    x = SqueezeExcitation1DLayer(out_dim=32, ratio=2, layer_name='se_2')(conv9_2)
    
    # MLP
    x = Flatten()(x)
    x = Dense(256)(x)
    x = tf.keras.activations.relu(x, alpha=0.1)
    x = Dropout(0.2)(x)
    x = Dense(128)(x)
    x = tf.keras.activations.relu(x, alpha=0.1)
    x = Dropout(0.2)(x)
    x = Dense(64)(x)
    x = tf.keras.activations.relu(x, alpha=0.1)
    x = Dropout(0.1)(x)
    x = Dense(32)(x)
    x = tf.keras.activations.relu(x, alpha=0.1)
    x = Dropout(0.1)(x)
    outputs = Dense(1, activation="linear")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    return model