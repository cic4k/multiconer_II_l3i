from tqdm import tqdm
IN_FILE = "/home/emiliano/Documents/MultiCoNER_2_data/ZH-Chinese/zh_test.conll"

def process_phrases(lines):

    phrases = []

    phrase_id = ""
    domain = ""
    phrase_text = []
    phrase_labels = []
    word_indexes = []
    for line in tqdm(lines):
        fields = line.split()
        if "# id " in line:
            index = 0
            changed_indexes = []

            corrupted = False
            tmp = line.split("\t")
            if len(tmp) == 3:
                phrase_id, domain, changed = tmp
            else :
                phrase_id= line.split(" ")[-1].replace("\n","")
                domain = ""
                changed = ""
            phrase_id = phrase_id.split(" ")[-1]
            if "corrupt=True" in domain:
                corrupted = True
                changed_indexes = changed.replace("\n","").split("=")[-1].replace("[","").replace("]","").split(",")
                changed_indexes = [_ for _ in map(int, changed_indexes)]

        elif len(fields) > 2:
            word, label = fields[0], fields[-1]
            word_index = index
            phrase_text.append(word)
            phrase_labels.append(label)
            if label != "O":
                word_indexes.append(word_index)
            index += 1
        elif len(phrase_id) != 0:
            phrase = {
                "phrase_id": phrase_id,
                "domain": domain,
                "phrase_text": phrase_text,
                "phrase_labels": phrase_labels,
                "corrputed": corrupted,
                "changed_indexes": changed_indexes,
                "word_indexes": word_indexes
            }
            phrases.append(phrase)
            phrase_id = ""
            domain = ""
            phrase_text = []
            phrase_labels = []
            word_indexes = []

    return phrases


with open(IN_FILE, 'r') as f:
    lines = f.readlines()

phrases = process_phrases(lines)
print()

total_changed = 0
nes_changed = 0
for phrase in phrases:
    if len(phrase["changed_indexes"])>0:
        total_changed += len(phrase["changed_indexes"])
        in_nes =list(set(phrase["changed_indexes"]).intersection(phrase["word_indexes"]))
        nes_changed += len(in_nes)

print(total_changed)
print(nes_changed)
