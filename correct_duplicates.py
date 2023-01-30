from tqdm import tqdm

REF = "/home/emiliano/Documents/MultiCoNER_2_data/EN-English/en_test.conll"
PRED = "./outputs/test_phase/bigbird-roberta-base-finetuned-multiconer2comp_autov2_pred.conll"
OUT = "./outputs/test_phase/bigbird-roberta-base-finetuned-multiconer2comp_autov2_pred2.conll"


with open(REF, "r") as ap:
    ref_content = ap.readlines()

ids = [line for line in ref_content if "# id" in line]

phrases = dict()
with open(PRED, "r") as ap:
    pred_content = ap.readlines()

phrase = []
for line in tqdm(pred_content):
    #line = line.replace("\n","")
    if "# id" in line:
        id = line.replace("	domain=en", "")
    elif line != "\n":
        phrase.append(line)
    elif line == "\n" and len(phrase)>0:
        phrases[id] = phrase
        phrase = []

with open(OUT, "w") as ap:

    for id in tqdm(ids):
        ap.write(f"{id}")
        ap.writelines(phrases[id])
        ap.write("\n")

print()