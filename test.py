import time

text = 'HelloWorld'

for char in text:
    print('\b', char, sep='', end='', flush=True)
    time.sleep(0.5)
