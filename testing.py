import itertools

a = 'alma'

b = [['alma'], [], ['körte', 'alma']]


def comp(t1, t2):
    return t1 in t2


l = [x for x in itertools.starmap(comp, zip(itertools.repeat(a), b))]
print(l)
