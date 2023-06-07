from tqdm import tqdm

REF = "/home/emiliano/Documents/MultiCoNER_2_data/MULTI_Multilingual/multi_dev.conll"
PRED = "/home/emiliano/PycharmProjects/multiconer_II_l3i/outputs/dev_phase/multi/checkpoint-11294_pred.conll"
OUT =  "/home/emiliano/PycharmProjects/multiconer_II_l3i/outputs/dev_phase/multi/checkpoint-11294_pred2.conll"



with open(REF, "r") as ap:
    ref_content = ap.readlines()

#ids = [line.replace("\tdomain=zh", "") for line in ref_content if "# id" in line]
ids = [line.split("\t")[0] for line in ref_content if "# id" in line]

phrases = dict()
with open(PRED, "r") as ap:
    pred_content = ap.readlines()

phrase = []
for line in tqdm(pred_content):
    #line = line.replace("\n","")
    if "# id" in line:
        #id = line.replace("\tdomain=en", "")
        id = line.split("\t")[0]
    elif line != "\n":
        phrase.append(line)
    elif line == "\n" and len(phrase)>0:
        phrases[id] = phrase
        phrase = []

with open(OUT, "w") as ap:

    for id in tqdm(ids):
        ap.write(f"{id}\n")
        ap.writelines(phrases[id])
        ap.write("\n\n")

print()