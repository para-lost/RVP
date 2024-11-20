import csv
import json
from shutil import ReadError


def test_nextqa():
    letter_list = ['A', 'B', 'C', 'D', 'E']
    filename = 'your-csv-file'
    results = []
    tot_num = 0
    acc = 0
    counter = 0 
    with open(filename, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            counter += 1
            result = row['result'].split("###")[0]
            answer = int(row['answer'])
            letter_answer = letter_list[answer]
            if 'None' not in result:
                tot_num += 1              
            else:
                print(result)
            if letter_answer in result:
                acc += 1
            else:
                pass           
    print(acc/counter)
    print(tot_num)


test_nextqa()


