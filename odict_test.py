from collections import OrderedDict

d = OrderedDict()

d["a"] = 2
d["b"] = 3
d["c"] = 1

print(d)


sorted_tuples = sorted(d.items(), key=lambda x: x[1], reverse=True)

ids = [t[0] for t in sorted_tuples]
scores = [t[1] for t in sorted_tuples]


print(ids)
print(scores)
