def learn(concepts,target):
    for i in range(len(concepts)):
        if target[i]=="Yes":
            spec=target[i]
            break
    gen = [["?" for i in range(len(spec))] for i in range (len(spec))]
    for i,h in enumerate(concepts):
        if target[i]=="Yes":
            for x in range(len(spec)):
                if h[x] !=spec[x]:
                    spec[x]="?"
                    gen[x][x]="?"            
        if target[i]=="No":
            for x in range(len(spec)): 
                if h[x]!= spec[x]:                    
                    gen[x][x] = spec[x]                
                else:                    
                    gen[x][x] = '?'
    ind = [i for i,val in enumerate(gen) if val == ["????"]]
    for i in ind:   
        gen.remove(['?', '?', '?', '?']) 
    return spec, gen 