import itertools
from transformers import AutoTokenizer
from transformers import AutoModelForMaskedLM
import os
import torch
import csv
import copy
from utils import NES
from tqdm import tqdm

PATH = "./MultiCoNER_2_data/train_dev_test"
LANG = ["multi"]
#LANG = ["bn","de","en","es","fa","fr","hi","it","pt","sv","uk","zh"]
#MODE = ["train", "dev"]
MODE = ["test"]
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
#MASKED_LM = "google/bigbird-roberta-base"
MASKED_LM = "xlm-roberta-base"

BASE_TEMPLATE = "[MASK] is the [ENTITY] ?"

complete_dataset = True
automatic = True

def process_phrases(lines):

    phrases = []

    phrase_id = ""
    domain = ""
    phrase_text = []
    phrase_labels = []

    for line in tqdm(lines):
        fields = line.split()
        if "# id " in line:
            if "domain=" in line:
                phrase_id, domain = line.split("\t")
                phrase_id = phrase_id.split(" ")[-1]
                domain = domain.split("=")[-1].replace("\n", "")
            else:
                phrase_id = line.split(" ")[-1].replace("\n","")
                domain = "test"
        elif len(fields) > 2:
            word, label = fields[0], fields[-1]
            phrase_text.append(word)
            phrase_labels.append(label)
        elif len(phrase_id) != 0:
            phrase = {
                "phrase_id": phrase_id,
                "domain": domain,
                "phrase_text": phrase_text,
                "phrase_labels": phrase_labels
            }
            phrases.append(phrase)
            phrase_id = ""
            domain = ""
            phrase_text = []
            phrase_labels = []

    return phrases


def create_templates():

    templates = list()
    templates_keys = list()

    if automatic:
        print(f"Devices: {torch.cuda.device_count()}")
        tokenizer = AutoTokenizer.from_pretrained(MASKED_LM)
        #model = AutoModelForMaskedLM.from_pretrained(MASKED_LM, attention_type="original_full")
        model = AutoModelForMaskedLM.from_pretrained(MASKED_LM)

        device = torch.device("cuda")
        model.to(device)

        print(f"{MASKED_LM} tokenizer & model loaded !")

        mask_token = tokenizer.mask_token

        for ne in NES.keys():
            text_entity = NES[ne][0]
            templates_keys.append(ne)

            template = BASE_TEMPLATE.replace("[ENTITY]", text_entity).replace("[MASK]", mask_token)
            templates.append(template)
        tokenized_templates = tokenizer(templates, return_tensors='pt', max_length=128, padding="max_length")
        tokenized_templates = tokenized_templates.to(device)

        mask_token_index = torch.where(tokenized_templates.data["input_ids"] == tokenizer.mask_token_id)[1]

        output = model(**tokenized_templates).logits
        mask_token_logits = []

        for _ in range(len(templates)):
            tmp = output[_, mask_token_index[_], :]
            tmp = torch.reshape(tmp, (1, -1))
            mask_token_logits.append(tmp)

        mask_token_logits = torch.cat(mask_token_logits, dim=0)
        mask_token_logits = mask_token_logits[:, None, :]
        mask_token_logits = torch.softmax(mask_token_logits, dim=2)
        mask_token_ids = torch.argmax(mask_token_logits, dim=2)
        mask_token_texts = tokenizer.decode(torch.reshape(mask_token_ids, (-1,))).split()

        for _, template_key in enumerate(templates_keys):
            NES[template_key][0] = templates[_].replace(mask_token, mask_token_texts[_])

    else:
        for ne in NES.keys():
            text_entity = NES[ne][0]
            NES[ne][0] = BASE_TEMPLATE.replace("[ENTITY]", text_entity).replace("[MASK]", NES[ne][1])


def build_questions(phrases, complete_dataset=False):

    for phrase in tqdm(phrases):
        entities_span = list()
        ent_start = ent_end = None
        for _, label in enumerate(phrase["phrase_labels"]):
            if label.startswith("B"):
                if ent_start is not None:
                    entities_span.append([ent_start, ent_end + 1])
                ent_start = ent_end = _
            elif label.startswith("I"):
                ent_end = _
            elif (label.startswith("O") or label.startswith("_"))  and ent_start is not None:
                entities_span.append([ent_start, ent_end+1])
                ent_start = ent_end = None
            if _ == len(phrase["phrase_labels"])-1 and ent_start is not None:
                entities_span.append([ent_start, ent_end+1])
        phrase["entities_span"] = entities_span

    for phrase in tqdm(phrases):
        questions = list()
        answers = list()
        entities_type = list()
        answers_start = list()
        local_NES = copy.deepcopy(NES)
        for _, entity_span in enumerate(phrase["entities_span"]):
            start, end = entity_span
            entity_type = phrase["phrase_labels"][start].replace("B-", "")
            local_NES.pop(entity_type, None)
            question = NES[entity_type][0]
            answer = " ".join(phrase["phrase_text"][start:end])

            entities_type.append(entity_type)
            questions.append(question)
            answers.append(answer)
            char_start = 0
            for i in range(start):
                char_start += len(phrase["phrase_text"][i])+1
            answers_start.append(char_start)

        if complete_dataset:
            for entity_type in local_NES.keys():
                question = local_NES[entity_type][0]
                answer = "None"
                answer_start = -1

                questions.append(question)
                answers.append(answer)
                entities_type.append(entity_type)
                answers_start.append(answer_start)

        phrase["questions"] = questions
        phrase["answers"] = answers
        phrase["entities_type"] = entities_type
        phrase["answers_start"] = answers_start
        phrase["phrase_text"] = " ".join(phrase["phrase_text"])

def write_tsv(phrases, output_file):

    id_phrase = 1
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["id", "id_multiconer", "domain", "context", "question", "answer",
                                                     "answer_start", "answer_type"], dialect="unix", delimiter="\t")
        writer.writeheader()
        for phrase in tqdm(phrases):
            entry = {
                "id_multiconer": phrase["phrase_id"],
                "domain": phrase["domain"],
                "context": phrase["phrase_text"],
            }
            for _, question in enumerate(phrase["questions"]):
                entry["id"] = f"p{id_phrase}"
                entry["question"] = question
                entry["answer"] = phrase["answers"][_]
                entry["answer_start"] = phrase["answers_start"][_]
                entry["answer_type"] = phrase["entities_type"][_]
                writer.writerow(entry)
                id_phrase += 1

def main():

    create_templates()
    torch.cuda.empty_cache()


    combinations = [_ for _ in itertools.product(*[LANG, MODE])]

    """
    for _lang, _mode in combinations:
        input_file = os.path.join(PATH, _lang + '-' + _mode + '.conll')
        with open(input_file, 'r') as f:
            lines = f.readlines()

        phrases = process_phrases(lines)
        build_questions(phrases)

        write_tsv(phrases, input_file.replace(".conll", ".tsv"))
    """
    if complete_dataset:
        for _lang, _mode in combinations:
            input_file = os.path.join(PATH, _lang + '-' + _mode + '.conll')
            with open(input_file, 'r') as f:
                lines = f.readlines()

            phrases = process_phrases(lines)
            build_questions(phrases, complete_dataset=complete_dataset)

            write_tsv(phrases, input_file.replace(".conll", "_comp.tsv"))

if __name__ == '__main__':
    """
    Starts the whole app from the command line
    """

    main()
