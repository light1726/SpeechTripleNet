import tensorflow as tf


class CNEN:
    class Train:
        random_seed = 123
        epochs = 1000
        train_batch_size = 32
        test_batch_size = 8
        test_interval = 100
        shuffle_buffer = 128
        shuffle = True
        num_samples = 1
        length_weight = 1.
        content_kl_weight = 1e-2
        spk_kl_weight = 1e-5
        learning_rate = 1.25e-4

    class Dataset:
        buffer_size = 65536
        num_parallel_reads = 64
        pad_factor = 0  # factor ** (num_blk - 1)
        dev_set_rate = 0.05
        test_set_rate = 0.05
        segment_size = 16

    class Texts:
        pad = '_'
        bos = '^'
        eos = '~'
        characters = '_^~$%12345abcdefghijklmnopqrstuvwxyz '

    class Audio:
        num_mels = 80
        num_freq = 1025
        min_mel_freq = 0.
        max_mel_freq = 8000.
        sample_rate = 16000
        frame_length_sample = 800
        frame_shift_sample = 200
        preemphasize = 0.97
        min_level_db = -100.0
        ref_level_db = 20.0
        max_abs_value = 1
        symmetric_specs = False
        griffin_lim_iters = 60
        power = 1.5
        center = True
        preprocess_n_jobs = 16

    class Common:
        latent_dim = 128
        output_dim = 80
        reduction_factor = 2
        mel_text_len_ratio = 4.

    class Decoder:
        class Transformer:
            pre_n_conv = 2
            pre_conv_kernel = 3
            pre_drop_rate = 0.2
            nblk = 2
            attention_dim = 256
            attention_heads = 4
            ffn_hidden = 1024
            attention_temperature = 1.
            attention_causality = False
            attention_window = 16
            post_n_conv = 5
            post_conv_filters = 256
            post_conv_kernel = 5
            post_drop_rate = 0.2

    class Posterior:
        class Transformer:
            pre_n_conv = 2
            pre_conv_kernel = 3
            pre_hidden = 256
            pre_drop_rate = 0.2
            pos_drop_rate = 0.2
            nblk = 2
            attention_dim = 256
            attention_heads = 4
            temperature = 1.0
            attention_causality = False
            attention_window = 8
            ffn_hidden = 1024


class ESDHPS:
    class Train:
        random_seed = 123
        epochs = 1000
        train_batch_size = 32
        test_batch_size = 8
        test_interval = 100
        shuffle_buffer = 128
        shuffle = True
        num_samples = 1
        content_kl_weight = 2e-2
        spk_kl_weight = 1e-5
        learning_rate = 1.25e-4

    class Dataset:
        buffer_size = 65536
        num_parallel_reads = 64
        segment_size = 16
        pad_factor = 0  # factor ** (num_blk - 1)

    class Texts:
        pad = '_'
        bos = '^'
        eos = '~'
        characters = '_^~abcdefghijklmnopqrstuvwxyz!\'\"(),-.:;? []'

    class Audio:
        num_mels = 80
        num_freq = 1025
        min_mel_freq = 0.
        max_mel_freq = 8000.
        sample_rate = 16000
        frame_length_sample = 800
        frame_shift_sample = 200
        preemphasize = 0.97
        min_level_db = -100.0
        ref_level_db = 20.0
        max_abs_value = 1
        symmetric_specs = False
        griffin_lim_iters = 60
        power = 1.5
        center = True
        preprocess_n_jobs = 16
        sil_trim_db = 30.

    class Common:
        latent_dim = 128
        output_dim = 80
        reduction_factor = 2
        mel_text_len_ratio = 4.

    class Encoder:
        class Transformer:
            vocab_size = 43
            embd_dim = 512
            n_conv = 3
            pre_hidden = 512
            conv_kernel = 5
            pre_activation = tf.nn.relu
            pre_drop_rate = 0.1
            pos_drop_rate = 0.1
            bn_before_act = False
            n_blk = 4
            attention_dim = 256
            attention_heads = 4
            attention_temperature = 1.0
            attention_causality = False
            attention_window = -1
            ffn_hidden = 1024

    class Decoder:
        class Transformer:
            pre_n_conv = 2
            pre_conv_kernel = 3
            pre_drop_rate = 0.2
            nblk = 2
            attention_dim = 256
            attention_heads = 4
            ffn_hidden = 1024
            attention_temperature = 1.
            attention_causality = False
            attention_window = 16
            post_n_conv = 5
            post_conv_filters = 256
            post_conv_kernel = 5
            post_drop_rate = 0.2

    class Posterior:
        class Transformer:
            pre_n_conv = 2
            pre_conv_kernel = 3
            pre_hidden = 256
            pre_drop_rate = 0.2
            pos_drop_rate = 0.2
            nblk = 2
            attention_dim = 256
            attention_heads = 4
            temperature = 1.0
            attention_causality = False
            attention_window = 8
            ffn_hidden = 1024

    class Prior:
        class Transformer:
            n_blk = 6
            n_transformer_blk = 2
            attention_dim = 256
            attention_heads = 4
            temperature = 1.0
            attention_causality = False
            attention_window = -1
            ffn_hidden = 1024
            inverse = False

    class SpeakerPosterior:
        class LSTMSpeakerEmbd:
            n_layer = 3
            lstm_hidden = 512

        class SelfAttSpkEmbd:
            pre_hidden = 256
            pre_drop_rate = 0.2
            pos_drop_rate = 0.2
            pre_activation = tf.nn.relu
            nblk = 2
            attention_dim = 256
            attention_heads = 4
            temperature = 1.0
            attention_causality = False
            attention_window = -1
            ffn_hidden = 1024
