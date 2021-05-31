from tqdm import tqdm
import tensorflow as tf

# Dataset file path
BASEPATH = "./dataset/"
DATASET = {
    "ncbi": {
        "from": {
            "train": "NCBI/NCBItrainset_corpus.txt",
            "dev": "NCBI/NCBIdevelopset_corpus.txt",
            "test": "NCBI/NCBItestset_corpus.txt",
        },
        "to": {
            "train": "NCBI/train.txt",
            "dev": "NCBI/dev.txt",
            "test": "NCBI/test.txt",
            "zs_test": "NCBI/zs_test.txt",
        },
    },
    "cdr": {
        "from": {
            "train": "CDR/CDR_TrainingSet.PubTator.txt",
            "dev": "CDR/CDR_DevelopmentSet.PubTator.txt",
            "test": "CDR/CDR_TestSet.PubTator.txt",
        },
        "to": {
            "train": "CDR/train.txt",
            "dev": "CDR/dev.txt",
            "test": "CDR/test.txt",
            "zs_test": "CDR/zs_test.txt",
        },
    },
}

one_hot = lambda y, len_tag: tf.one_hot(tf.cast(y, tf.int32), len_tag)
compute_ner_loss = lambda label, logits: tf.reduce_sum(
    -tf.reduce_sum(label * tf.math.log(logits), axis=-1)
)
compute_nen_loss = lambda label, logits: -tf.reduce_sum(label * tf.math.log(logits))


@tf.function
def train_one_step(
    model, optimizer, s_ind, s_seg, e_ind, e_seg, ner, cpt_ner, nen, MU, LAMBDA
):
    with tf.GradientTape() as tape:
        logits_ner, logits_cpt_ner, logits_nen = model(s_ind, s_seg, e_ind, e_seg)
        ner_loss = compute_ner_loss(ner, logits_ner)
        cpt_ner_loss = compute_ner_loss(cpt_ner, logits_cpt_ner)
        nen_loss = compute_nen_loss(nen, logits_nen)
        loss = MU * ner_loss + cpt_ner_loss + LAMBDA * nen_loss

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    return loss


def evaluate(model, dataLoader, dtype="train"):
    real_len = []
    ner_label = []
    ner_logits = []
    cpt_ner_label = []
    cpt_ner_logits = []
    nen_label = []
    nen_logits = []
    for s_ind, s_seg, e_ind, e_seg, ner, cpt_ner, nen, seq_len in tqdm(
        dataLoader.Data(dtype), ascii=True
    ):
        l_ner, l_cpt_ner, l_nen = model(s_ind, s_seg, e_ind, e_seg)
        ner_label.append(ner)
        ner_logits.append(l_ner)
        cpt_ner_label.append(cpt_ner)
        cpt_ner_logits.append(l_cpt_ner)
        nen_label.append(nen)
        nen_logits.append(l_nen)
        real_len.append(seq_len)
    real_len = tf.concat(real_len, axis=0)
    label_ner = tf.concat(ner_label, axis=0)
    logits_ner = tf.concat(ner_logits, axis=0)
    label_cpt_ner = tf.concat(cpt_ner_label, axis=0)
    logits_cpt_ner = tf.concat(cpt_ner_logits, axis=0)
    label_nen = tf.concat(nen_label, axis=0)
    logits_nen = tf.concat(nen_logits, axis=0)

    cpt_ner_prec, cpt_ner_reca, cpt_ner_f1 = dataLoader.evaluate_ner(
        logits_cpt_ner, label_cpt_ner, real_len
    )
    nen_prec, nen_reca, nen_f1 = dataLoader.evaluate_nen(
        logits_ner,
        label_ner,
        logits_cpt_ner,
        label_cpt_ner,
        real_len,
        logits_nen,
        label_nen,
    )
    return {
        "ner": [cpt_ner_prec, cpt_ner_reca, cpt_ner_f1],
        "nen": [nen_prec, nen_reca, nen_f1],
    }


def save_prediction_result(model, dataLoader, file_name, dtype="train"):
    n_samples = 0
    with open(file_name, "w", encoding="utf-8") as fp:
        for s_ind, s_seg, e_ind, e_seg, ner, _, nen, _ in tqdm(
            dataLoader.Data(dtype), ascii=True
        ):
            p_ner, p_cpt_ner, p_nen, attn = model(
                s_ind, s_seg, e_ind, e_seg, return_attention=True
            )
            p_cpt_ner = tf.argmax(p_cpt_ner, axis=-1)
            p_ner = tf.argmax(p_ner, axis=-1)
            p_nen = tf.argmax(p_nen, axis=-1)
            for i in range(len(s_ind)):
                sentence = dataLoader.parse_idx_tokens(s_ind.numpy().tolist()[i])
                entity = dataLoader.parse_idx_tokens(e_ind.numpy().tolist()[i])
                ner_labels = dataLoader.parse_idx_ner_labels(
                    ner.numpy().tolist()[i][1 : len(sentence) + 1]
                )
                pred_ner_labels = dataLoader.parse_idx_ner_labels(
                    p_ner.numpy().tolist()[i][1 : len(sentence) + 1]
                )
                pred_cpt_ner_labels = dataLoader.parse_idx_ner_labels(
                    p_cpt_ner.numpy().tolist()[i][1 : len(sentence) + 1]
                )
                attention = attn.numpy().tolist()[i][1 : len(sentence) + 1]

                # ID
                fp.write(f"{n_samples}\t")
                # Entity
                fp.write(f"{' '.join(entity)}\t")
                # NEN
                fp.write(f"NEN:{int(nen.numpy()[i])}\tPred:{int(p_nen.numpy()[i])}\n")
                # NER
                tmp = ""
                for t_s, t_ner, t_pner, t_cptner, t_attn in zip(
                    sentence,
                    ner_labels,
                    pred_ner_labels,
                    pred_cpt_ner_labels,
                    attention,
                ):
                    tmp += f"{t_s}\t{t_ner}\t{t_pner}\t{t_cptner}\t{t_attn}\n"
                fp.write(f"{tmp}\n")
                n_samples += 1


def file_log(path, string):
    with open(path, "a") as fp:
        fp.write(string)
        fp.write("\n")