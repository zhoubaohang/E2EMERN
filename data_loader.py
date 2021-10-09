import copy
import codecs
from operator import getitem
import numpy as np
from tqdm import trange
import tensorflow as tf
from utils import BASEPATH
from typing import List, Dict
from conlleval import evaluate
from keras_bert import Tokenizer
from tensorflow.data import Dataset
from entitybase.entity_base_loader import EntityBase
from keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import precision_score, recall_score, f1_score


class DataLoader(object):
    def __init__(self, bert_path: str, dict_dataset: Dict):

        self.__bert_path = f"{bert_path}/vocab.txt"
        self.__tokenizer = self.__load_vocabulary()
        self.__dict_dataset = dict_dataset
        self.__entity_base = EntityBase(bert_path)
        self.__dict_ner_label = ["X"]

        self.__max_seq_len = 100
        self.__max_ent_len = 16
        self.__batch_size = 6

        self._train_data = self.__parse_data(f"{BASEPATH}{dict_dataset['train']}")
        self._devel_data = self.__parse_data(f"{BASEPATH}{dict_dataset['dev']}")
        self._test_data = self.__parse_data(f"{BASEPATH}{dict_dataset['test']}")
        self._zs_test_data = self.__parse_data(f"{BASEPATH}{dict_dataset['zs_test']}")

    def resampling_data(self, dtype):
        if dtype == "train":
            self._train_data = self.__parse_data(
                f"{BASEPATH}{self.__dict_dataset['train']}"
            )
        elif dtype == "devel":
            self._devel_data = self.__parse_data(
                f"{BASEPATH}{self.__dict_dataset['dev']}"
            )
        elif dtype == "test":
            self._test_data = self.__parse_data(
                f"{BASEPATH}{self.__dict_dataset['test']}"
            )

    def parse_idx_tokens(self, ind):
        return self.__tokenizer.decode(ind)

    def parse_idx_ner_labels(self, ner):
        return [self.__dict_ner_label[i] for i in ner]

    def __parse_idx_sequence(self, pred, label):
        res_pred, res_label = [], []
        records = {}

        for i in range(len(pred)):
            tmp_pred, tmp_label = [], []
            str_pred = " ".join([str(ele) for ele in pred[i].numpy().tolist()])
            str_label = " ".join([str(ele) for ele in label[i].numpy().tolist()])
            str_record = str_pred + str_label
            if str_record in records:
                tmp = records[str_record]
                res_pred.append(tmp[0])
                res_label.append(tmp[1])

            else:
                for p, l in zip(pred[i], label[i]):
                    if self.__dict_ner_label[l] != "X":
                        tmp_label.append(self.__dict_ner_label[l])

                        if self.__dict_ner_label[p] == "X":
                            tmp_pred.append("O")
                        else:
                            tmp_pred.append(self.__dict_ner_label[p])
                res_pred.append(tmp_pred)
                res_label.append(tmp_label)
                records[str_record] = (tmp_pred, tmp_label)

        return res_pred, res_label

    def evaluate_ner(self, logits, label, real_len):
        pred = tf.argmax(logits, axis=-1)
        pred, true = self.__parse_idx_sequence(pred, label)
        y_real, pred_real = [], []
        records = []
        for i in trange(len(real_len), ascii=True):
            record = " ".join(true[i]) + str(real_len[i])
            if record not in records:
                records.append(record)
                y_real.extend(true[i][1 : 1 + real_len[i]])
                pred_real.extend(pred[i][1 : 1 + real_len[i]])
        prec, rec, f1 = evaluate(y_real, pred_real, verbose=False)
        return (prec / 100, rec / 100, f1 / 100)

    def __restore_ner_label(self, ner_logits, ner_label, real_len):
        ner_pred = tf.argmax(ner_logits, axis=-1)
        ner_pred, ner_truth = self.__parse_idx_sequence(ner_pred, ner_label)
        ner_label_real, ner_pred_real = [], []
        for i in range(len(real_len)):
            ner_label_real.append(ner_truth[i][1 : 1 + real_len[i]])
            ner_pred_real.append(ner_pred[i][1 : 1 + real_len[i]])
        return ner_label_real, ner_pred_real

    def __extract_entity_by_index(self, label_sequence, index):
        length = len(label_sequence)
        entity = []
        tmp_index = index
        while (
            tmp_index >= 0 and tmp_index < length and label_sequence[tmp_index] != "O"
        ):
            entity.insert(0, tmp_index)
            if "B-" in label_sequence[tmp_index]:
                break
            tmp_index -= 1
        tmp_index = index + 1
        while tmp_index < length and label_sequence[tmp_index] != "O":
            if "B-" in label_sequence[tmp_index]:
                break
            entity.append(tmp_index)
            tmp_index += 1
        return entity

    def evaluate_nen(
        self,
        ner_logits,
        ner_label,
        cpt_ner_logits,
        cpt_ner_label,
        real_len,
        nen_logits,
        nen_label,
    ):
        ner_label_real, ner_pred_real = self.__restore_ner_label(
            ner_logits, ner_label, real_len
        )
        cpt_ner_label_real, cpt_ner_pred_real = self.__restore_ner_label(
            cpt_ner_logits, cpt_ner_label, real_len
        )

        nen_pred = tf.argmax(nen_logits, axis=-1).numpy().tolist()
        nen_label = nen_label.numpy().tolist()
        tmp_nen_pred = []
        tmp_nen_label = []
        for i in trange(len(nen_label), ascii=True):
            n_entity = 0
            if nen_label[i] == 1:
                for e in ner_label_real:
                    if "B-" in e:
                        n_entity += 1
                tmp_nen_label.extend([1] * n_entity)
            else:
                tmp_nen_label.append(0)

            if nen_pred[i] == 0:
                tmp_nen_pred.extend([0] * n_entity)
            else:
                index = 0
                flag = False
                for p, t in zip(ner_pred_real[i], ner_label_real[i]):
                    if "B-" in p or "I-" in p:
                        if p == t:
                            if not flag:
                                if self.__extract_entity_by_index(
                                    cpt_ner_pred_real[i], index
                                ) == self.__extract_entity_by_index(
                                    cpt_ner_label_real[i], index
                                ):
                                    tmp_nen_pred.append(1)
                                else:
                                    tmp_nen_pred.append(0)
                                flag = True
                        else:
                            if not flag:
                                tmp_nen_pred.append(0)
                                flag = True
                    else:
                        flag = False
                    index += 1

            if len(tmp_nen_label) < len(tmp_nen_pred):
                size = len(tmp_nen_label)
                for _ in range(len(tmp_nen_pred) - size):
                    tmp_nen_label.append(nen_label[i])
            elif len(tmp_nen_label) > len(tmp_nen_pred):
                size = len(tmp_nen_pred)
                for _ in range(len(tmp_nen_label) - size):
                    tmp_nen_pred.append(0)

        filtered_nen_label, filtered_nen_pred = [], []
        for i in range(len(tmp_nen_label)):
            if tmp_nen_label[i] == 0 and tmp_nen_pred[i] == 0:
                continue
            filtered_nen_label.append(tmp_nen_label[i])
            filtered_nen_pred.append(tmp_nen_pred[i])
        reca = recall_score(filtered_nen_label, filtered_nen_pred, average="weighted")
        prec = precision_score(
            filtered_nen_label, filtered_nen_pred, average="weighted"
        )
        f1 = f1_score(filtered_nen_label, filtered_nen_pred, average="weighted")
        return (prec, reca, f1)

    @property
    def LABEL_SIZE(self):
        return len(self.__dict_ner_label)

    def Data(self, dtype: str):
        return getattr(self, f"_{dtype}_data")

    def __load_vocabulary(self):
        token_dict = {}

        with codecs.open(self.__bert_path, "r", "utf8") as reader:
            for line in reader:
                token = line.strip()
                token_dict[token] = len(token_dict)

        return Tokenizer(token_dict)

    def __tokenize_entity(self, entities):
        indices = []
        segments = []

        for e in entities:
            ind, seg = self.__tokenizer.encode(first=e)
            indices.append(ind)
            segments.append(seg)

        indices = pad_sequences(indices, self.__max_ent_len, value=0, padding="post")
        segments = pad_sequences(segments, self.__max_ent_len, value=0, padding="post")

        return (indices, segments)

    def __tokenize_sample(self, sentence, label, cpt_label):
        labels = []
        cpt_labels = []
        indices = []
        segments = []

        for i in range(len(sentence)):
            s = sentence[i][: self.__max_seq_len - 2]
            tmp_indice, tmp_segment, tmp_label, tmp_cpt_label = [], [], [], []
            CLS_idx, SEP_idx = 0, 0

            for j, w in enumerate(s):
                ind, seg = self.__tokenizer.encode(first=w)
                if j == 0:
                    CLS_idx, SEP_idx = ind[0], ind[-1]
                ind = ind[1:-1]
                seg = seg[1:-1]

                for k in range(len(ind)):
                    if k == 0:
                        tmp_label.append(self.__dict_ner_label.index(label[i][j]))
                        tmp_cpt_label.append(
                            self.__dict_ner_label.index(cpt_label[i][j])
                        )
                    else:
                        tmp_label.append(self.__dict_ner_label.index("X"))
                        tmp_cpt_label.append(
                            self.__dict_ner_label.index(cpt_label[i][j])
                        )

                tmp_indice.extend(ind)
                tmp_segment.extend(seg)

            tmp_indice = tmp_indice[: self.__max_seq_len - 2]
            tmp_indice = [CLS_idx] + tmp_indice + [SEP_idx]

            tmp_segment = tmp_segment[: self.__max_seq_len - 2]
            tmp_segment = [0] + tmp_segment + [0]

            tmp_label = tmp_label[: self.__max_seq_len - 2]
            tmp_label = (
                [self.__dict_ner_label.index("O")]
                + tmp_label
                + [self.__dict_ner_label.index("O")]
            )

            tmp_cpt_label = tmp_cpt_label[: self.__max_seq_len - 2]
            tmp_cpt_label = (
                [self.__dict_ner_label.index("O")]
                + tmp_cpt_label
                + [self.__dict_ner_label.index("O")]
            )

            indices.append(tmp_indice)
            segments.append(tmp_segment)
            labels.append(tmp_label)
            cpt_labels.append(tmp_cpt_label)

        labels = pad_sequences(
            labels,
            self.__max_seq_len,
            value=self.__dict_ner_label.index("O"),
            padding="post",
        )
        cpt_labels = pad_sequences(
            cpt_labels,
            self.__max_seq_len,
            value=self.__dict_ner_label.index("O"),
            padding="post",
        )
        indices = pad_sequences(indices, self.__max_seq_len, value=0, padding="post")
        segments = pad_sequences(segments, self.__max_seq_len, value=0, padding="post")

        return (indices, segments, labels, cpt_labels)

    def __parse_data(self, path: str):
        data = self.__get_sentences(path)
        sentences = data["sentences"]
        ner_label = data["ner"]
        nen_label = data["nen"]

        parsed_real_len = []
        parsed_ner_label = []
        parsed_cpt_ner_label = []
        parsed_nen_label = []
        parsed_sent_indices = []
        parsed_sent_segments = []
        parsed_ent_indices = []
        parsed_ent_segments = []

        for i in trange(len(sentences), ascii=True):
            sentence, ner, nen = [ele[i] for ele in [sentences, ner_label, nen_label]]
            samples = self.__extend_sample(sentence, ner, nen, 1, "test" in path)

            pkd_sentence = samples["sentences"]
            pkd_ner_tags = samples["ner"]
            pkd_cpt_ner_tags = samples["cpt_ner"]
            pkd_nen_tags = samples["nen"]
            pkd_ent_sents = samples["ent_sents"]

            (
                sent_indices,
                sent_segments,
                t_ner_label,
                t_cpt_ner_label,
            ) = self.__tokenize_sample(pkd_sentence, pkd_ner_tags, pkd_cpt_ner_tags)
            parsed_sent_indices.append(sent_indices)
            parsed_sent_segments.append(sent_segments)
            parsed_ner_label.append(t_ner_label)
            parsed_cpt_ner_label.append(t_cpt_ner_label)

            parsed_nen_label.extend(pkd_nen_tags)
            ent_indices, ent_segments = self.__tokenize_entity(pkd_ent_sents)
            parsed_ent_indices.append(ent_indices)
            parsed_ent_segments.append(ent_segments)

            parsed_real_len.extend([len(sentence)] * len(sent_indices))

        parsed_ner_label = np.vstack(parsed_ner_label)
        parsed_cpt_ner_label = np.vstack(parsed_cpt_ner_label)
        parsed_sent_indices = np.vstack(parsed_sent_indices)
        parsed_sent_segments = np.vstack(parsed_sent_segments)
        parsed_ent_indices = np.vstack(parsed_ent_indices)
        parsed_ent_segments = np.vstack(parsed_ent_segments)

        dataset = Dataset.from_tensor_slices(
            (
                parsed_sent_indices,
                parsed_sent_segments,
                parsed_ent_indices,
                parsed_ent_segments,
                parsed_ner_label,
                parsed_cpt_ner_label,
                parsed_nen_label,
                parsed_real_len,
            )
        )
        dataset = (
            dataset.shuffle(len(parsed_real_len))
            .batch(self.__batch_size)
            .prefetch(tf.data.experimental.AUTOTUNE)
            .cache()
        )
        return dataset

    def __extend_sample(
        self,
        sentence: List[int],
        ner_tag: List[int],
        nen_tag: Dict,
        n_neg: int,
        flag: bool = False,
    ) -> Dict:
        """
        extend the sample to all samples with specific nen tags
        """
        pkd_ner_tags = []
        pkd_nen_tags = []
        pkd_sentences = []
        pkd_ent_sents = []
        pkd_cpt_ner_tags = []

        for src_ent, _ in nen_tag.items():
            tmp = ner_tag.copy()
            pkd_cpt_ner_tags.append(ner_tag.copy())

            for tar_ent, tar_idxs in nen_tag.items():
                if tar_ent != src_ent:
                    for begin, length in tar_idxs:
                        for i in range(begin, begin + length):
                            tmp[i] = "O"

            pkd_ner_tags.append(tmp)
            pkd_sentences.append(sentence.copy())

            ent_sent = self.__entity_base.getItem(src_ent)
            pkd_ent_sents.append(ent_sent)
            pkd_nen_tags.append(1)
        # Sampling negative entities
        if flag:
            cands = self.__entity_base.generate_candidates(
                sentence, list(nen_tag.keys())
            )
            for c in cands:
                pkd_ner_tags.append(["O"] * len(ner_tag))
                pkd_cpt_ner_tags.append(ner_tag.copy())
                pkd_sentences.append(sentence.copy())
                ent_sent = self.__entity_base.getItem(c)
                pkd_ent_sents.append(ent_sent)
                pkd_nen_tags.append(0)
        else:
            for i in range(n_neg):
                pkd_ner_tags.append(["O"] * len(ner_tag))
                pkd_cpt_ner_tags.append(ner_tag.copy())
                pkd_sentences.append(sentence.copy())
                ent_sent = self.__entity_base.random_entity(list(nen_tag.keys()))
                pkd_ent_sents.append(ent_sent)
                pkd_nen_tags.append(0)

        return {
            "ner": pkd_ner_tags,
            "cpt_ner": pkd_cpt_ner_tags,
            "nen": pkd_nen_tags,
            "sentences": pkd_sentences,
            "ent_sents": pkd_ent_sents,
        }

    def __parse_nen_tag(self, nen_tag: List[str]) -> Dict:
        """
        parse the nen tag sequence to the dictionary
        example:
            [1,1,-1,-1] -> {
                1: [(0,2)]
            }
        """
        tmp = []
        result = {}

        for i in range(len(nen_tag)):
            if nen_tag[i] != "O":
                tmp.append(i)
            else:
                if len(tmp):
                    cache = result.get(nen_tag[i], [])
                    cache.append((tmp[0], len(tmp)))
                    result[nen_tag[i - 1]] = cache
                    tmp.clear()

        return result

    def __get_sentences(self, path: str) -> Dict:
        """
        handle the input data files to the list form
        """
        ner_tags = []
        nen_tags = []
        sentences = []

        with open(path, "r") as fp:
            ner_tag = []
            nen_tag = []
            sentence = []

            for line in fp.readlines():
                line = line.strip()

                if line:
                    word, r_tag, *n_tag = line.split("\t")
                    sentence.append(word)
                    ner_tag.append(r_tag)
                    nen_tag.append(n_tag[0])

                    if "train" in path and r_tag not in self.__dict_ner_label:
                        self.__dict_ner_label.append(r_tag)
                else:
                    sentences.append(copy.deepcopy(sentence))
                    ner_tags.append(copy.deepcopy(ner_tag))
                    nen_tags.append(self.__parse_nen_tag(nen_tag))
                    sentence.clear()
                    ner_tag.clear()
                    nen_tag.clear()

        return {"sentences": sentences, "ner": ner_tags, "nen": nen_tags}
