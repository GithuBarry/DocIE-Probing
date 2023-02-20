import shutil,os,os.path

for file in os.listdir(".") :
  le = len("output_sentence")
  if file[:le] == "output_sentence":
    shutil.move(file, "./outputs")
  
