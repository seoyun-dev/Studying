def datacat(L):
    result=[]
    L_set=set(L)
    for l in L_set:
        result.append((l,L.count(l)))
    return result

L=['car', 6, 2, 7.5, 'banana', 'car', 6, 5]
print(datacat(L)) 

L=['cat','cat','cat','dog','dog',3,3,3]
print(datacat(L)) 