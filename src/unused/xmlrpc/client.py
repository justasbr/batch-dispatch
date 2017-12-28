import xmlrpc.client
import time

s = xmlrpc.client.ServerProxy('http://localhost:8000')
print(s.pow(2, 3))  # Returns 2**3 = 8
print(s.add(2, 3))  # Returns 5
print(s.mul(5, 2))  # Returns 5*2 = 10

s.call_func(lambda x: print(x), 22)

# Print list of available methods
print(s.system.listMethods())

# t1 = time.time()
# avg = []
# for i in range(1000):
#     if not i %100:
#         print(i, "something")
#     s.add(5, 2)
#     # print(s.mul(5, 2))  # Returns 5*2 = 10
#     t2 = time.time()
#     avg.append(t2-t1)
#     # print("ms since last", round(1000 * (t2 - t1), 2))
#     t1 = t2
# print(sum(avg)*1.0 / len(avg))