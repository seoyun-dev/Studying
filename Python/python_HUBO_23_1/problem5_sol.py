def Joseph(N,K):
    L=list(range(N))
    result=[]
    n=0
    while len(L)>0:
        n=(n+K-1)%len(L)
        result.append(L.pop(n))
    return result

print(Joseph(9,5))

