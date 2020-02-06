import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

class RegFHVAEnew(tf.keras.Model):
    ''' combine the encoder and decoder into an end-to-end model for training '''

    def __init__(self, z1_dim=32, z2_dim=32, z1_rhus=[256, 256], z2_rhus=[256, 256], x_rhus=[256, 256],
                 nmu2=5000, z1_nlabs={}, z2_nlabs={}, mu_nl=None,logvar_nl=None, tr_shape=(20, 80),
                 alpha_dis=1.0, alpha_reg_b=1.0, alpha_reg_c=1.0, name="autoencoder", **kwargs):

        super(RegFHVAEnew, self).__init__(name=name, **kwargs)

        self.tr_shape = tr_shape

        #encoder/decoder arch
        self.z1_dim, self.z2_dim = z1_dim, z2_dim
        self.z1_rhus, self.z2_rhus = z1_rhus, z2_rhus
        self.x_rhus = x_rhus

        #non linearities
        self.mu_nl, self.logvar_nl = mu_nl, logvar_nl

        #nlabs = dictionary with for each label, the dimension of the regularization vector
        self.z1_nlabs, self.z2_nlabs = z1_nlabs, z2_nlabs
        self.nmu2 = nmu2
        self.mu2_table = tf.Variable(tf.random.normal([nmu2, z2_dim], stddev=1.0))

        #loss factors
        self.alpha_dis = alpha_dis
        self.alpha_reg_b, self.alpha_reg_c = alpha_reg_b, alpha_reg_c

        #init net
        self.encoder = Encoder(self.z1_dim, self.z2_dim, self.z1_rhus, self.z2_rhus, self.tr_shape, self.mu_nl, self.logvar_nl)
        self.decoder = Decoder(self.x_rhus, self.tr_shape, self.mu_nl, self.logvar_nl)
        self.regulariser = Regulariser(self.z1_nlabs, self.z2_nlabs)

        #log-prior stddevs
        self.pz1_stddev = 1.0
        self.pz2_stddev = 0.5
        self.pmu2_stddev = 1.0

    def call(self, x, y):

        mu2 = tf.gather(self.mu2_table, y)

        z1_mu, z1_logvar, z1_sample, z2_mu, z2_logvar, z2_sample, qz1_x, qz2_x \
            = self.encoder(x)

        out, x_mu, x_logvar, x_sample, px_z\
            = self.decoder(x, z1_sample, z2_sample)

        z1_rlogits, z2_rlogits = self.regulariser(z1_mu, z2_mu)

        return mu2, qz2_x, z2_sample, qz1_x, z1_sample, px_z, x_sample, z1_rlogits, z2_rlogits

    def compute_loss(self, x, y, n, bReg, cReg, mu2, qz2_x, z2_sample, qz1_x, z1_sample, px_z, x_sample, z1_rlogits, z2_rlogits):

        # priors
        pz1 = [0., np.log(self.pz1_stddev ** 2).astype(np.float32)]
        pz2 = [mu2, np.log(self.pz2_stddev ** 2).astype(np.float32)]
        pmu2 = [0., np.log(self.pmu2_stddev ** 2).astype(np.float32)]

        # variational lower bound
        log_pmu2 = log_normal_pdf(mu2, pmu2[0], pmu2[1], raxis=1)
        log_px_z = log_normal_pdf(x, px_z[0], px_z[1], raxis=(1, 2))
        neg_kld_z2 = -1 * tf.reduce_sum(kld(qz2_x[0], qz2_x[1], pz2[0], pz2[1]), axis=1)
        neg_kld_z1 = -1 * tf.reduce_sum(kld(qz1_x[0], qz1_x[1], pz1[0], pz1[1]), axis=1)

        lb = log_px_z + neg_kld_z1 + neg_kld_z2 + log_pmu2 / n

        # discriminative loss
        fn = tf.nn.sparse_softmax_cross_entropy_with_logits

        logits = tf.expand_dims(qz2_x[0], 1) - tf.expand_dims(self.mu2_table, 0)
        logits = -1 * tf.pow(logits, 2) / (2 * tf.exp(pz2[1]))
        logits = tf.reduce_sum(input_tensor=logits, axis=-1)
        log_qy = -fn(labels=y, logits=logits)

        # Regularization loss, minimize x-entropy, maximize log_c
        # self.log_c = tf
        # for i, name in enumerate(nlabs.keys()):
        #     tf.stack((self.log_c,-tf.nn.softmax_cross_entropy_with_logits \
        #         (labels=tf.slice(self.cReg,[0,i],[-1,1]), logits=self.rlogits[i])),axis=0)

        TensorList = [tf.expand_dims(-fn(labels=tf.squeeze(tf.slice(bReg, [0, i], [-1, 1]), axis=-1), \
                                         logits=z1_rlogits[i]), axis=1) for i, name in enumerate(self.z1_nlabs.keys())]
        log_b = tf.concat(TensorList, axis=1)

        TensorList = [tf.expand_dims(-fn(labels=tf.squeeze(tf.slice(cReg, [0, i], [-1, 1]), axis=-1), \
                                         logits=z2_rlogits[i]), axis=1) for i, name in enumerate(self.z2_nlabs.keys())]
        log_c = tf.concat(TensorList, axis=1)

        log_b_loss = tf.reduce_sum(input_tensor=log_b, axis=1)
        log_c_loss = tf.reduce_sum(input_tensor=log_c, axis=1)

        loss = -1 * tf.reduce_mean(input_tensor= lb + self.alpha_dis * log_qy \
                + self.alpha_reg_b * log_b_loss + self.alpha_reg_c * log_c_loss)

        return loss, log_pmu2, neg_kld_z2, neg_kld_z1, log_px_z, lb, log_qy, log_b_loss, log_c_loss


class Encoder(layers.Layer):
    ''' encodes the input to the latent factors z1 and z2'''

    def __init__(self, z1_dim=32, z2_dim=32, z1_rhus=[256,256], z2_rhus=[256,256],
                 tr_shape=(20,80), mu_nl=None, logvar_nl=None, name="encoder", **kwargs):

        super(Encoder, self).__init__(name=name,   **kwargs)

        # latent dims
        self.z1_dim = z1_dim
        self.z2_dim = z2_dim

        # RNN specs for z2_pre_encoder
        self.z2_rhus = z2_rhus

        self.cells_z2 = [layers.LSTMCell(rhu) for rhu in self.z2_rhus]
        self.cell_z2 = layers.StackedRNNCells(self.cells_z2)
        # init_state= cell.get_initial_state(batch_size=tr_shape[0], dtype=x.dtype)
        self.fullRNN_z2 = layers.RNN(self.cell_z2, return_state=True, time_major=False)


        # RNN specs for z1_pre_encoder
        self.z1_rhus = z1_rhus

        self.cells_z1 = [layers.LSTMCell(rhu) for rhu in self.z1_rhus]
        self.cell_z1 = layers.StackedRNNCells(self.cells_z1)
        # init_state = cell.get_initial_state(batch_size=bs, dtype=x.dtype)
        self.fullRNN_z1 = layers.RNN(self.cell_z1, return_state=True, time_major=False)


        # fully connected layers for computation of mu and logvar
        self.z1mu_fclayer = layers.Dense(
            z1_dim, activation=mu_nl, use_bias=True,
            kernel_initializer='glorot_uniform', bias_initializer='zeros')

        self.z1logvar_fclayer = layers.Dense(
            z1_dim, activation=logvar_nl, use_bias=True,
            kernel_initializer='glorot_uniform', bias_initializer='zeros')

        self.z2mu_fclayer = layers.Dense(
            z2_dim, activation=mu_nl, use_bias=True,
            kernel_initializer='glorot_uniform', bias_initializer='zeros')

        self.z2logvar_fclayer = layers.Dense(
            z2_dim, activation=logvar_nl, use_bias=True,
            kernel_initializer='glorot_uniform', bias_initializer='zeros')


    def call(self, inputs):
        ''' The full decoder '''
        z2_pre_out = self.z2_pre_encoder(inputs)

        z2_mu = self.z2mu_fclayer(z2_pre_out)
        z2_logvar = self.z2logvar_fclayer(z2_pre_out)
        z2_sample = reparameterize(z2_mu, z2_logvar)

        z1_pre_out = self.z1_pre_encoder(inputs, z2_sample)

        z1_mu = self.z1mu_fclayer(z1_pre_out)
        z1_logvar = self.z1logvar_fclayer(z1_pre_out)
        z1_sample = reparameterize(z1_mu, z1_logvar)

        qz1_x = [z1_mu, z1_logvar]
        qz2_x = [z2_mu, z2_logvar]

        return z1_mu, z1_logvar, z1_sample, z2_mu, z2_logvar, z2_sample, qz1_x, qz2_x

    def z2_mu_separate(self, inputs):
        ''' To calculate mu2 for the lookup table '''
        z2_pre_out = self.z2_pre_encoder(inputs)
        z2_mu = self.z2mu_fclayer(z2_pre_out)
        return z2_mu

    def z2_pre_encoder(self, x):
        """
        Pre-stochastic layer encoder for z2 (latent sequence variable)
        Args:
            x(tf.Tensor): tensor of shape (bs, T, F)
        Return:
            out(tf.Tensor): concatenation of hidden states of all LSTM layers
        """
        outputs = self.fullRNN_z2(inputs=x, training=True)  #, initial_state=init_state)

        # if z2_rhus = [16,32] and bs = 256 and we use return_state=True, then outputs is a list containing:
        # [ Tensor(256x32), [Tensor(256x16), Tensor(256x16)], [Tensor(256x32),Tensor(256x32)] ]
        # and we concatenate the second one (=h, first one is cell state c) of both layers
        out = [outputs[i+1][1] for i in range(len(self.z2_rhus))]  #range starts with 0 and stops before end

        out = tf.concat(out, axis=-1)
        return out

    def z1_pre_encoder(self, x, z2):
        """
        Pre-stochastic layer encoder for z1 (latent segment variable)
        Args:
            x(tf.Tensor): tensor of shape (bs, T, F)
            z2(tf.Tensor): tensor of shape (bs, D1)
        Return:
            out(tf.Tensor): concatenation of hidden states of all LSTM layers
        """

        bs, T = tf.shape(input=x)[0], tf.shape(input=x)[1]
        z2 = tf.tile(tf.expand_dims(z2, 1), (1, T, 1))
        x_z2 = tf.concat([x, z2], axis=-1)

        # see z2_pre_encoder for details about RNN
        outputs = self.fullRNN_z1(inputs=x_z2, training=True)  #, initial_state=init_state)
        out = [outputs[i+1][1] for i in range(len(self.z2_rhus))]
        out = tf.concat(out, axis=-1)

        return out


class Decoder(layers.Layer):
    ''' decodes factors z1 and z2 to reconstructed input x'''

    def __init__(self, x_rhus=[256,256], tr_shape=(20, 80),
                 mu_nl=None, logvar_nl=None, name="decoder", **kwargs):

        super(Decoder, self).__init__(name=name,   **kwargs)

        # x
        self.tr_shape = tr_shape

        # RNN specs
        self.x_rhus = x_rhus

        self.cells_x = [layers.LSTMCell(rhu) for rhu in self.x_rhus]
        self.cell_x = layers.StackedRNNCells(self.cells_x)
        #init_state = cell.get_initial_state(batch_size=bs, dtype=x.dtype)
        self.fullRNN_x = layers.RNN(self.cell_x, return_sequences=True, time_major=False)


        # fully connected layers for computing mu and logvar
        self.xmu_fclayer = layers.Dense(
            tr_shape[1], activation=mu_nl, use_bias=True,
            kernel_initializer='glorot_uniform', bias_initializer='zeros')

        self.xlogvar_fclayer = layers.Dense(
            tr_shape[1], activation=logvar_nl, use_bias=True,
            kernel_initializer='glorot_uniform', bias_initializer='zeros')


    def call(self, x, z1, z2):

        bs, T = tf.shape(input=x)[0], tf.shape(input=x)[1]

        z1 = tf.tile(tf.expand_dims(z1, 1), (1, T, 1))
        z2 = tf.tile(tf.expand_dims(z2, 1), (1, T, 1))
        z1_z2 = tf.concat([z1, z2], axis=-1)

        # return_sequences=True returns the entire sequence of outputs for each sample
        # (one vector per timestep per sample)
        # shape of output is (batch_size, timesteps, units)
        # so this is what we need: the output of the RNN for every timestep
        output = self.fullRNN_x(inputs=z1_z2, training=True)  #, initial_state=init_state)

        x_mu, x_logvar, x_sample = [], [], []
        for timestep in range(0, T):
            out_t = output[:, timestep, :]
            x_mu_t = self.xmu_fclayer(out_t)
            x_logvar_t = self.xlogvar_fclayer(out_t)
            x_sample_t = reparameterize(x_logvar_t, x_mu_t)

            x_mu.append(x_mu_t)
            x_logvar.append(x_logvar_t)
            x_sample.append(x_sample_t)

        x_mu = tf.stack(x_mu, axis=1)
        x_logvar = tf.stack(x_logvar, axis=1)
        x_sample = tf.stack(x_sample, axis=1)
        px_z = [x_mu, x_logvar]

        return output, x_mu, x_logvar, x_sample, px_z

class Regulariser(layers.Layer):
    ''' predicts the regularizing factors '''

    def __init__(self, z1_nlabs={}, z2_nlabs={}, name="regulariser", **kwargs):

        super(Regulariser, self).__init__(name=name, **kwargs)

        #dict with e.g. [('gender',3), ('region',9)]
        self.z1_nlabs = z1_nlabs
        self.z2_nlabs = z2_nlabs

        self.z1_nlabs_per_fac = list(self.z1_nlabs.values())
        self.z2_nlabs_per_fac = list(self.z2_nlabs.values())

        self.reg_z1_fclayer = layers.Dense(
            sum(self.z1_nlabs_per_fac), activation=None, use_bias=True,
            kernel_initializer='glorot_uniform', bias_initializer='zeros')

        # sum minus 1?
        self.reg_z2_fclayer = layers.Dense(
            sum(self.z2_nlabs_per_fac), activation=None, use_bias=True,
            kernel_initializer='glorot_uniform', bias_initializer='zeros')

    def call(self, z1_mu, z2_mu):

        z1_all_rlogits = self.reg_z1_fclayer(z1_mu)
        z2_all_rlogits = self.reg_z2_fclayer(z2_mu)

        z1_rlogits = self.fix_logits(z1_all_rlogits, self.z1_nlabs_per_fac)
        z2_rlogits = self.fix_logits(z2_all_rlogits, self.z2_nlabs_per_fac)

        return z1_rlogits, z2_rlogits

    def fix_logits(self, all_rlogits, nlabs_per_fac):
        # split, pad, concat and put in list all the logits
        rlogits = []

        for tens in tf.split(all_rlogits, nlabs_per_fac, 1):
            T = tf.shape(input=tens)[0]
            z = tf.zeros([T, 1], dtype=tf.float32)
            # add column of zeros at start
            #rlogits.append(tf.concat((z, tens), axis=1))
            rlogits.append(tens)

        return rlogits


#@tf.function
def log_normal_pdf(x, mu=0., logvar=0., raxis=1):
  log2pi = tf.math.log(2. * np.pi)
  return tf.reduce_sum(
          -.5 * ((x - mu) ** 2. * tf.exp(-logvar) + logvar + log2pi),
          axis=raxis)

#@tf.function
def reparameterize(mu, logvar):
    eps = tf.random.normal(shape=mu.shape)
    return eps * tf.exp(logvar * .5) + mu

#@tf.function
def kld(p_mu, p_logvar, q_mu, q_logvar):
    """
    compute D_KL(p || q) of two Gaussians
    """
    # Added extra brackets after the minus sign
    return -0.5 * (1 + p_logvar - q_logvar - \
                   (((p_mu - q_mu) ** 2 + tf.exp(p_logvar)) / tf.exp(q_logvar)))
