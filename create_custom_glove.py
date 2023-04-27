import os
import pandas as pd
from dankware import multithread, clr, cls

cls()

try: os.chdir(os.path.dirname(__file__))
except: pass

comments = pd.read_csv('train.csv')['comment_text'].values

words = {}
for comment in comments:
    for split in comment.split(" "):
        if not split in words.keys():
            words[split] = ''
del comments

words = sorted(set(list(words.keys())))

open("words.txt", "w+", encoding="utf8").write("\n".join(words))
words = open("words.txt", "r", encoding="utf8").read().split("\n")
words = sorted(set(words))
open("words.txt", "w+", encoding="utf8").write("\n".join(words))

print(clr(f"\n  > Total words: {len(words)}"))
del words

def find_str_in_file(search_string):
    with open("words.txt", "r", encoding="utf8") as file:
        for line in file:
            if search_string in line:
                return True
    return False

def check_and_save(line):
    if find_str_in_file(line.split(" ")[0]):
        optimised_glove_lines.append(line)

glove_name = "glove.840B.300d.txt"
glove_lines = []
optimised_glove_lines = []

with open(glove_name, "r", encoding="utf8") as file:
    for line in file:
        glove_lines.append(line)
        if len(glove_lines) == 10000:
            multithread(check_and_save, 2000, glove_lines)
            glove_lines = []
            open("optimised_" + glove_name, "a+", encoding="utf8").write("".join(optimised_glove_lines))
            optimised_glove_lines = []

