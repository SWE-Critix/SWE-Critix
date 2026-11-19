import json
import shutil
import os
from datasets import Dataset
import re
import datasets

def write_list_to_disk_as_json(path, data):
    """
    Save a list as a json file to path
    https://stackoverflow.com/questions/52137608/writing-a-large-json-array-to-file
    """
    print("Writing a list to a json file " + path)
    with open(path, "w") as jsonFile:
        for item in data:
            json.dump(item, jsonFile)
            jsonFile.write('\n')

def write_list_to_disk_as_json_in_chinese(path, data, mode="w"):
    """
    Save a list as a json file to path, the language is chinese
    """
    print("Writing a list to a json file " + path)
    with open(path, mode, encoding="utf8") as jsonFile:
        for item in data:
            json.dump(item, jsonFile, ensure_ascii=False)
            jsonFile.write('\n')

def read_json_object_from_disk(path):
    """
    Read a json object from a json file
    """
    print("Reading a json object from a json file " + path)
    with open(path, "r", encoding="utf8") as jsonFile:
        data = json.load(jsonFile)
        return data

def write_json_object_to_disk(path, data):
    """
    Write a json object to a json file
    """
    print("Writing a json object to a json file " + path)
    with open(path, "w", encoding="utf8") as jsonFile:
        json.dump(data, jsonFile, ensure_ascii=False)


def read_json_list_from_disk(path):
    """
    Read a list from a json file
    https://stackoverflow.com/questions/12451431/loading-and-parsing-a-json-file-with-multiple-json-objects
    """
    print("Reading a list from a json file " + path)
    data = []
    with open(path, "r", encoding="utf8") as jsonFile:
        for line in jsonFile:
            data.append(json.loads(line))
        return data

def clear_and_create_dir(dir_path):
    shutil.rmtree(dir_path, ignore_errors=True)  # clean folder
    os.makedirs(dir_path)  # create folder

def create_dir(dir_path):
    os.makedirs(dir_path)  # create folder

def write_list_to_disk_as_hf_dataset(dir_path, data):
    """
    Construct a huggingface dataset from a list and save to file
    """
    print("Creating a huggingface dataset from a list")
    dataset = Dataset.from_list(data)
    print("Saving a huggingface dataset to " + dir_path)
    dataset.save_to_disk(dir_path)

def is_empty_string(s):
    pattern = re.compile(r'\s+')
    return re.sub(pattern, '', s) == ""

def concatenate_datasets_from_chunk_folders(root_path):
    folder_names = os.listdir(root_path)
    num_folders = len(folder_names)
    chunks = []
    for i in range(1, num_folders + 1):
        # use number to iterate because if use folders to iterate, the order will be chunk1, chunk10, chunk11, chunk12, ..., chunk19, chunk2, chunk20, ...
        ds_chunk = datasets.load_from_disk(os.path.join(root_path, "chunk" + str(i)))
        chunks.append(ds_chunk)

    return datasets.concatenate_datasets(chunks)

def flatten(xss):
    return [x for xs in xss for x in xs]