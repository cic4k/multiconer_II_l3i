from datasets import Dataset, DatasetDict
import pandas
import os

NES = {
    #Location
    "Facility": ["facility", "where"],
    "HumanSettlement": ["human settlement", "where"],
    "Station": ["station", "where"],
    "OtherLOC": ["general location", "what"],

    #Creative works
    "VisualWork": ["visual work", "what"],
    "MusicalWork": ["musical work", "what"],
    "WrittenWork": ["written work", "what"],
    "ArtWork": ["work of art", "what"],
    "Software": ["software", "what"],
    #OtherCW

    #Group
    "MusicalGRP": ["music group", "who"],
    "PublicCorp": ["public corporation", "who"],
    "PrivateCorp": ["private corporation", "who"],
    #OtherCorp
    "AerospaceManufacturer": ["aerospace manufacturer", "who"],
    "SportsGRP": ["sport team", "who"],
    "CarManufacturer": ["car manufacturer", "who"],
    #TechCORP
    "ORG": ["general organization", "what"],

    #Person
    "Scientist": ["scientist", "who"],
    "Artist": ["artist", "who"],
    "Athlete": ["athlete", "who"],
    "Politician": ["politician", "who"],
    "Cleric": ["cleric", "who"],
    "SportsManager": ["sport manager", "who"],
    "OtherPER": ["general person", "what"],

    #Product
    "Clothing": ["clothing", "which"],
    "Vehicle": ["vehicle", "which"],
    "Food": ["food", "which"],
    "Drink": ["drink", "which"],
    "OtherPROD": ["general product", "what"],

    #Medical
    "Medication/Vaccine": ["medication or vaccine", "what"],
    "MedicalProcedure": ["medical procedure", "what"],
    "AnatomicalStructure": ["anatomical structure", "what"],
    "Symptom": ["medical symptom", "what"],
    "Disease": ["disease", "what"]
}

def load_multiconer_dataset(dataset_path, lang):

    file_path = os.path.join(dataset_path, f"{lang}-train_comp.tsv")
    dataset_train = Dataset.from_pandas(pandas.read_csv(file_path, sep="\t", header=0))

    file_path = os.path.join(dataset_path, f"{lang}-dev_comp.tsv")
    dataset_dev = Dataset.from_pandas(pandas.read_csv(file_path, sep="\t", header=0))

    return DatasetDict({
        "train": dataset_train,
        "development": dataset_dev
    })
