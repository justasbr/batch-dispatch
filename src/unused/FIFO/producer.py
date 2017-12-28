import time


def produce():
    i = 0
    while True:
        time.sleep(0.1)
        i += 1
        yield 'Message' + str(i)

prod = produce()
for x in prod:
    print(x)
