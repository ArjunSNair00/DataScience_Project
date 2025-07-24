import os
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, "Number.txt")
file = open(file_path, "r")
print("Prime numbers: ")
for number in file:
  flag=0
  num=number.strip(" ")
  num=int(num)
  for i in range(2,num):
    if num%i==0:
      flag=1
      break
  if flag==0:
    print(num)