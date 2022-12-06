import os
from argparse import ArgumentParser
from argparse import ArgumentDefaultsHelpFormatter
from transformers import AutoTokenizer
from transformers import AutoModelForQuestionAnswering, TrainingArguments, Trainer
from transformers import default_data_collator
from tqdm import tqdm
import utils
import numpy as np
import collections
from pprint import pprint

os.environ["WANDB_DISABLED"] = "true"
#os.environ["CUDA_VISIBLE_DEVICES"] = "2,1,3"

def parse_arguments():

    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

    #QA dataset params
    parser.add_argument("-p", "--dataset_path",
                        dest="dataset_path",
                        default="./MultiCoNER_2_train_dev/train_dev/",
                        type=str,
                        help="""QA dataset folder""")

    parser.add_argument("-g", "--lang",
                        dest="lang",
                        default="en",
                        type=str,
                        help="""Language for QA dataset""")

    #models
    parser.add_argument("-q", "--model",
                        dest="model",
                        default="xlm-roberta-base",
                        type=str,
                        help="""Pretrained QA model""")


    #tokenizer params
    parser.add_argument("-m", "--max_length",
                        dest="max_length",
                        default=512,
                        type=int,
                        help="""Maximum length of a feature (question and context), 512 for xlm-roberta-base""")

    parser.add_argument("-s", "--doc_stride",
                        dest="doc_stride",
                        default=128,
                        type=int,
                        help="""Authorized overlap between two part of the context when splitting it is needed""")

    #training/test params
    parser.add_argument("-b", "--batch_size_train",
                        dest="batch_size_train",
                        default=4,
                        type=int,
                        help="""Batch size""")

    parser.add_argument("-t", "--batch_size_test",
                        dest="batch_size_test",
                        default=16,
                        type=int,
                        help="""Batch size""")

    parser.add_argument("-e", "--epochs",
                        dest="epochs",
                        default=4,
                        type=int,
                        help="""Number of epochs""")

    parser.add_argument("-r", "--mode",
                        dest="mode",
                        choices=["train", "test", "all"],
                        default="all",
                        type=str,
                        help="""Operation mode""")

    parser.add_argument("-x", "--suffix",
                        dest="suffix",
                        default="",
                        type=str,
                        help="""Suffix to output names""")

    return parser.parse_args()

def prepare_train_features(samples, lm_tokenizer, max_length, doc_stride):

    pad_on_right = lm_tokenizer.padding_side == "right"


    tokenized_samples = lm_tokenizer(
        samples["question" if pad_on_right else "context"],
        samples["context" if pad_on_right else "question"],
        truncation="only_second" if pad_on_right else "only_first",
        max_length=max_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length"
    )

    overflow_samples_mapping = tokenized_samples.pop("overflow_to_sample_mapping")
    offsets_mapping = tokenized_samples.pop("offset_mapping") #offset_mapping -> the offset of each token in the input string

    if "bigbird" in lm_tokenizer.name_or_path:
        samples_ids2tokens = [lm_tokenizer.convert_ids_to_tokens(_) for _ in tokenized_samples["input_ids"]]


    tokenized_samples["start_positions"] = []
    tokenized_samples["end_positions"] = []

    for i, offsets in enumerate(offsets_mapping):

        if "bigbird" in lm_tokenizer.name_or_path:
            sample_ids2tokens = samples_ids2tokens[i]
            for j, token in enumerate(sample_ids2tokens):
                if token[0] == "▁" and offsets[j][0] != 0:
                    offsets[j] = (offsets[j][0]+1, offsets[j][1])

        # Impossible answers will be labeled with the index of the CLS token
        sample_input_ids = tokenized_samples["input_ids"][i]
        cls_index = sample_input_ids.index(lm_tokenizer.cls_token_id)

        # Grab the sequence corresponding to that example (to know what is the context and what is the question)
        # [None, 0, 0, 0, 0, 0, None, None, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, None, None, None, None, None, ... ]
        sample_sequence_ids = tokenized_samples.sequence_ids(i)

        sample_index = overflow_samples_mapping[i]
        sample_answer = samples["answer"][sample_index]

        """
        Impossible answers : answer -> None, answer_start -> -1  
        """
        if sample_answer == "None":
            tokenized_samples["start_positions"].append(cls_index)
            tokenized_samples["end_positions"].append(cls_index)
        else:
            sample_answer_start_char = samples["answer_start"][sample_index]
            sample_answer_end_char = sample_answer_start_char + len(sample_answer)

            # Start token index of the current span in the text.
            sample_answer_token_start_index = 0
            while sample_sequence_ids[sample_answer_token_start_index] != (1 if pad_on_right else 0):
                sample_answer_token_start_index += 1

            # End token index of the current span in the text.
            sample_answer_token_end_index = len(sample_input_ids) - 1
            while sample_sequence_ids[sample_answer_token_end_index] != (1 if pad_on_right else 0):
                sample_answer_token_end_index -= 1

            # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
            if not (offsets[sample_answer_token_start_index][0] <= sample_answer_start_char and
                    offsets[sample_answer_token_end_index][1] >= sample_answer_end_char):
                tokenized_samples["start_positions"].append(cls_index)
                tokenized_samples["end_positions"].append(cls_index)
            else:
                # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                # Note: we could go after the last offset if the answer is the last word (edge case).
                while (sample_answer_token_start_index < len(offsets) and
                       offsets[sample_answer_token_start_index][0] <= sample_answer_start_char):
                    sample_answer_token_start_index += 1
                tokenized_samples["start_positions"].append(sample_answer_token_start_index - 1)

                while offsets[sample_answer_token_end_index][1] >= sample_answer_end_char:
                    sample_answer_token_end_index -= 1
                tokenized_samples["end_positions"].append(sample_answer_token_end_index + 1)

    return tokenized_samples

def prepare_test_features(samples, lm_tokenizer, max_length, doc_stride):

    pad_on_right = lm_tokenizer.padding_side == "right"

    samples["question"] = [q.lstrip() for q in samples["question"]]

    tokenized_samples = lm_tokenizer(
        samples["question" if pad_on_right else "context"],
        samples["context" if pad_on_right else "question"],
        truncation="only_second" if pad_on_right else "only_first",
        max_length=max_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    overflow_samples_mapping = tokenized_samples.pop("overflow_to_sample_mapping")

    if "bigbird" in lm_tokenizer.name_or_path:
        samples_ids2tokens = [lm_tokenizer.convert_ids_to_tokens(_) for _ in tokenized_samples["input_ids"]]

    # We keep the sample_id that gave us this feature and we will store the offset mappings.
    tokenized_samples["sample_id"] = []

    for i in range(len(tokenized_samples["input_ids"])):

        if "bigbird" in lm_tokenizer.name_or_path:
            sample_ids2tokens = samples_ids2tokens[i]
            for j, token in enumerate(sample_ids2tokens):
                if token[0] == "▁" and tokenized_samples["offset_mapping"][i][j][0] != 0:
                    tokenized_samples["offset_mapping"][i][j] = (tokenized_samples["offset_mapping"][i][j][0]+1,
                                                                 tokenized_samples["offset_mapping"][i][j][1])

        # Grab the sequence corresponding to that example (to know what is the context and what is the question).
        sequence_ids = tokenized_samples.sequence_ids(i)
        context_index = 1 if pad_on_right else 0

        # One example can give several spans, this is the index of the example containing this span of text.
        sample_index = overflow_samples_mapping[i]
        tokenized_samples["sample_id"].append(samples["id"][sample_index])

        # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
        # position is part of the context or not.
        tokenized_samples["offset_mapping"][i] = [
            (o if sequence_ids[k] == context_index else None)
            for k, o in enumerate(tokenized_samples["offset_mapping"][i])
        ]

    return tokenized_samples

def postprocess_qa_predictions(samples, lm_tokenizer, features, raw_predictions, n_best_size=10, max_answer_length=30):

    all_start_logits, all_end_logits = raw_predictions
    # Build a map example to its corresponding features.
    sample_id_to_index = {k: i for i, k in enumerate(samples["id"])}
    features_per_sample = collections.defaultdict(list)
    for i, feature in enumerate(features):
        #features_per_sample[i].append(i)
        features_per_sample[sample_id_to_index[feature["sample_id"]]].append(i)


    predictions = collections.OrderedDict()

    # Logging.
    print(f"Post-processing {len(samples)} example predictions split into {len(features)} features.")

    for sample_index, sample in enumerate(tqdm(samples)):

        # Those are the indices of the features associated to the current example.
        feature_indices = features_per_sample[sample_index]

        min_null_score = None
        valid_answers = []

        # Looping through all the features associated to the current sample.
        for feature_index in feature_indices:
            # We grab the predictions of the model for this feature.
            start_logits = all_start_logits[feature_index]
            end_logits = all_end_logits[feature_index]
            # This is what will allow us to map some the positions in our logits to span of texts in the original
            # context. Each element  is a tuple (start_char, end_char) of the tokenized context
            offset_mapping = features[feature_index]["offset_mapping"]

            # Update minimum null prediction.
            cls_index = features[feature_index]["input_ids"].index(lm_tokenizer.cls_token_id)
            feature_null_score = start_logits[cls_index] + end_logits[cls_index]
            if min_null_score is None or min_null_score < feature_null_score:
                min_null_score = feature_null_score

            # Go through all possibilities for the `n_best_size` greater start and end logits.
            start_indexes = np.argsort(start_logits)[-1: -n_best_size - 1: -1].tolist()
            end_indexes = np.argsort(end_logits)[-1: -n_best_size - 1: -1].tolist()
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # Don't consider out-of-scope answers, either because the indices are out of bounds or correspond
                    # to part of the input_ids that are not in the context.
                    if (
                            start_index >= len(offset_mapping)
                            or end_index >= len(offset_mapping)
                            or offset_mapping[start_index] is None
                            or offset_mapping[end_index] is None
                    ):
                        continue
                    # Don't consider answers with a length that is either < 0 or > max_answer_length.
                    if end_index < start_index or end_index - start_index + 1 > max_answer_length:
                        continue

                    start_char = offset_mapping[start_index][0]#get the start_char index from (start_char, end_char)
                    end_char = offset_mapping[end_index][1]#get the end_char index from (start_char, end_char)
                    valid_answers.append(
                        {
                            "score": start_logits[start_index] + end_logits[end_index],
                            "start_char": start_char,
                            "end_char": end_char
                        }
                    )

        if len(valid_answers) > 0:
            valid_answers.sort(key=lambda x: x["score"], reverse=True)
        else:
            # In the very rare edge case we have not a single non-null prediction, we create a fake prediction to avoid
            # failure.
            valid_answers.append({"start_char": -1, "end_char": -1, "score": 0.0})

        null_score = {
            "score": min_null_score,
            "is_best": True if min_null_score > valid_answers[0]["score"] else False
        }

        if null_score["is_best"]:
            _ = 0
            if valid_answers[_]["score"] <= 0.0:
                predictions.setdefault(sample["id_multiconer"], {})
                predictions[sample["id_multiconer"]][sample["answer_type"]] = [[-1, -1, null_score["score"]]]
            else:
                predictions.setdefault(sample["id_multiconer"], {})
                predictions[sample["id_multiconer"]][sample["answer_type"]] = list()
                while _ < len(valid_answers) and valid_answers[_]["score"] >= 0.0:

                    start_char = valid_answers[_]["start_char"]
                    end_char = valid_answers[_]["end_char"]
                    score = valid_answers[_]["score"]

                    if len(predictions[sample["id_multiconer"]][sample["answer_type"]]) > 0:
                        if not check_overlap([start_char, end_char], predictions[sample["id_multiconer"]][sample["answer_type"]]):
                            predictions[sample["id_multiconer"]][sample["answer_type"]].append([start_char, end_char, score])
                    else:
                        predictions[sample["id_multiconer"]][sample["answer_type"]].append([start_char, end_char, score])

                    _ += 1
        else:
            _ = 0
            predictions.setdefault(sample["id_multiconer"], {})
            predictions[sample["id_multiconer"]][sample["answer_type"]] = list()
            while _ < len(valid_answers) and valid_answers[_]["score"] >= null_score["score"]:

                start_char = valid_answers[_]["start_char"]
                end_char = valid_answers[_]["end_char"]
                score = valid_answers[_]["score"]

                if len(predictions[sample["id_multiconer"]][sample["answer_type"]]) > 0:
                    if not check_overlap([start_char, end_char], predictions[sample["id_multiconer"]][sample["answer_type"]]):
                        predictions[sample["id_multiconer"]][sample["answer_type"]].append([start_char, end_char, score])
                else:
                    predictions[sample["id_multiconer"]][sample["answer_type"]].append([start_char, end_char, score])

                _ += 1
        predictions[sample["id_multiconer"]]["context"] = sample["context"]

    return predictions


def check_overlap(actual, tmp_predictions):
    """
    Return the amount of overlap, in bp
    between a and b.
    If >0, the number of bp of overlap
    If 0,  they are book-ended.
    If <0, the distance in bp between them
    """

    a_start_char = actual[0]
    a_end_char = actual[1]

    for _ in tmp_predictions:
        p_start_char = _[0]
        p_end_char = _[1]

        if min(a_end_char, p_end_char) - max(a_start_char, p_start_char) >= 0:
            return True

    return False


def predictions2IOB(predictions):

    for key_pred, prediction in predictions.items():
        context = prediction["context"]

        tokens = []
        start_index = 0
        for end_index, char in enumerate(context):
            if char == " " or end_index == len(context)-1:
                if end_index == len(context)-1:
                    end_index += 1
                tokens.append({
                    "offset": [start_index, end_index],
                    "text": context[start_index:end_index],
                    "entities": []
                })
                start_index = end_index + 1

        for ne_type in utils.NES.keys():
            if prediction[ne_type][0][0] != -1:
                for valid_prediction in prediction[ne_type]:
                    pred_start_index = valid_prediction[0]
                    pred_end_index = valid_prediction[1]
                    pred_score = valid_prediction[2]
                    prefix = "B-"
                    for token in tokens:
                        start_index, end_index = token["offset"]
                        if min(end_index, pred_end_index) - max(start_index, pred_start_index) >= 0:
                            token["entities"].append([f"{prefix}{ne_type}", pred_score])
                            prefix = "I-"

        for token in tokens:
            token["entities"].sort(key=lambda x: x[-1], reverse=True)

        prediction["IOB"] = tokens

def writeIOB(predictions, out_file_name):

    out_string = "\n"
    for key_pred, prediction in predictions.items():
        out_string += f"# id {key_pred}	domain=en\n"
        t_entity_type_prev = ""
        for token in prediction["IOB"]:
            t_text = token["text"]
            t_entity = token["entities"][0][0] if len(token["entities"]) > 0 else "O"
            if t_entity != "O":
                t_entity_prefix, t_entity_type = t_entity.split("-")
                if t_entity_prefix == "I" and t_entity_type != t_entity_type_prev:
                    t_entity = "O"
                else:
                    t_entity_type_prev = t_entity_type

            line = f"{t_text} _ _ {t_entity}\n"
            out_string += line
        out_string += "\n\n"

    with open(out_file_name, 'w') as f:
        f.write(out_string)


def train_model(dataset, lm_tokenizer, args):

    if not os.path.isdir("./models"):
        os.mkdir("./models")

    tokenized_dataset = dataset.map(prepare_train_features,
                                    batched=True,
                                    remove_columns=dataset["train"].column_names,
                                    fn_kwargs={"lm_tokenizer": lm_tokenizer,
                                               "max_length": args.max_length,
                                               "doc_stride": args.doc_stride}
                                    )

    if "bigbird" in lm_tokenizer.name_or_path:
        model = AutoModelForQuestionAnswering.from_pretrained(args.model, attention_type="original_full")
    else:
        model = AutoModelForQuestionAnswering.from_pretrained(args.model)

    model_name = os.path.split(args.model)[-1]
    out_model_name = os.path.join("./models", f"{model_name}-finetuned-multiconer2comp_{args.suffix}")

    train_args = TrainingArguments(
        out_model_name,
        save_strategy="epoch",
        logging_strategy="epoch",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=args.batch_size_train,
        per_device_eval_batch_size=args.batch_size_test,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        push_to_hub=False
    )

    data_collator = default_data_collator

    trainer = Trainer(
        model,
        train_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["development"],
        data_collator=data_collator,
        tokenizer=lm_tokenizer,

    )

    trainer.train()
    trainer.save_model(out_model_name)

    return out_model_name


def test_model(dataset, lm_tokenizer, args):

    tokenized_test_dataset = dataset["development"].map(prepare_test_features,
                                                        batched=True,
                                                        remove_columns=dataset["development"].column_names,
                                                        fn_kwargs={"lm_tokenizer": lm_tokenizer,
                                                                   "max_length": args.max_length,
                                                                   "doc_stride": args.doc_stride}
                                                        )

    if "bigbird" in lm_tokenizer.name_or_path:
        model = AutoModelForQuestionAnswering.from_pretrained(args.model, attention_type="original_full")
    else:
        model = AutoModelForQuestionAnswering.from_pretrained(args.model)

    trainer_args = TrainingArguments(
        output_dir="./tmp_trainer",
        per_device_eval_batch_size=args.batch_size_test,
        push_to_hub=False,
    )

    data_collator = default_data_collator

    trainer = Trainer(
        model,
        trainer_args,
        data_collator=data_collator,
        tokenizer=lm_tokenizer,

    )
    raw_predictions = trainer.predict(tokenized_test_dataset)

    predictions = postprocess_qa_predictions(dataset["development"],
                                             lm_tokenizer,
                                             tokenized_test_dataset,
                                             raw_predictions.predictions)

    predictions2IOB(predictions)

    if not os.path.isdir("./outputs"):
        os.mkdir("./outputs")

    base_out_file_name = f"{os.path.split(args.model)[-1]}_pred"
    base_out_file_name = os.path.join("./outputs", base_out_file_name)
    with open(f"{base_out_file_name}.txt", mode="w") as file_object:
        pprint(predictions, stream=file_object)

    writeIOB(predictions, f"{base_out_file_name}.conll")

def main():
    args = parse_arguments()

    dataset = utils.load_multiconer_dataset(args.dataset_path, args.lang)

    lm_tokenizer = AutoTokenizer.from_pretrained(args.model)

    if args.mode == "all":
        out_model_name = train_model(dataset, lm_tokenizer, args)
        args.model = out_model_name
        test_model(dataset, lm_tokenizer, args)
    elif args.mode == "train":
        train_model(dataset, lm_tokenizer, args)
    elif args.mode == "test":
        test_model(dataset, lm_tokenizer, args)

if __name__ == '__main__':
    """
    Starts the whole app from the command line
    """

    main()
