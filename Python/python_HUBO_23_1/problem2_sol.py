def datacat2(L):
    result={}
    Num_to_Eng={1:'one', 2:'two', 3:'three', 4:'four', 5:'five'}
    for l in L:
        if len(l) in result:
            result[len(l)]+=1
        else:
            result[len(l)]=1
    for key in result:
        result[key]=Num_to_Eng[result[key]]
    return result

print(datacat2(['car','sun','son','dive','love','finalexam','python']))
print(datacat2(['information','superficial','great','nice','goodluck','happy']))
