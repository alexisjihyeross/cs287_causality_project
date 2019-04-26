import csv

#make binary versions of train and dev datasets (remove neutral examples)
def make_binary(folder, file):
  with open(folder + "/" + file) as in_file, open(folder + "/binary_" + file, mode="w") as out_file:
      reader = csv.reader(in_file, delimiter='\t', quoting=csv.QUOTE_NONE)
      writer = csv.writer(out_file, delimiter='\t')
      for row in reader:
          if row[-1] != "neutral":
              writer.writerow(row)


make_binary("glue_data/MNLI", "dev_matched.tsv")
make_binary("glue_data/MNLI", "dev_mismatched.tsv")
make_binary("glue_data/MNLI", "train.tsv")
