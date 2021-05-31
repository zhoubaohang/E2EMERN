import tensorflow as tf
from attention import AttentionWeightedAverage
from keras_bert import load_trained_model_from_checkpoint
from keras.layers import Dense, RepeatVector, GlobalAveragePooling1D


class E2EMERN(tf.keras.models.Model):
    def __init__(self, bert_path: str, n_label: int):
        super(E2EMERN, self).__init__()
        self.__n_label = n_label
        self.__ckpt_path = f"{bert_path}/bert_model.ckpt"
        self.__config_path = f"{bert_path}/bert_config.json"

        self.fc1 = Dense(2, activation="softmax")
        self.fc2 = Dense(self.__n_label, activation="softmax")
        self.fc3 = Dense(1024, activation="sigmoid")
        self.fc4 = Dense(1024, activation="elu")
        self.fc5 = Dense(self.__n_label, activation="softmax")
        self.attention = AttentionWeightedAverage(
            return_attention=True, name="entity_attention"
        )

        self.cls_feature = GlobalAveragePooling1D()
        self.bert_model = load_trained_model_from_checkpoint(
            self.__config_path, self.__ckpt_path, seq_len=None
        )
        for l in self.bert_model.layers:
            l.trainable = True

    @tf.function
    def call(self, s_ind, s_seg, e_ind, e_seg, return_attention=False):
        sent_feature = self.bert_model([s_ind, s_seg])
        seq_len = sent_feature.get_shape().as_list()[1]

        ent_feature = self.bert_model([e_ind, e_seg])

        # Low-level task
        logits_cpt_ner = self.fc5(sent_feature)

        # Entity Attention
        cls_ent_feature = self.cls_feature(ent_feature)

        h_star, attn = self.attention(
            tf.concat([sent_feature, RepeatVector(seq_len)(cls_ent_feature)], axis=-1)
        )
        h_star = self.fc4(h_star)

        # Mid-level task
        logits_nen = self.fc1(tf.concat([h_star, cls_ent_feature], axis=-1))

        # High-level task
        gate = self.fc3(
            tf.concat([sent_feature, RepeatVector(seq_len)(h_star)], axis=-1)
        )
        logits_ner = self.fc2(
            (1 - gate) * sent_feature + gate * RepeatVector(seq_len)(cls_ent_feature)
        )

        if return_attention:
            return logits_ner, logits_cpt_ner, logits_nen, attn
        else:
            return logits_ner, logits_cpt_ner, logits_nen
