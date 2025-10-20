def myFunc(e):
  a=len(e)
  return a

names = ['ArjAAun', 'RAa', 'AksAAAAhay', 'AbAh']

names.sort(key=myFunc)
print(names)