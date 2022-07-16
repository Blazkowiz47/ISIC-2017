with open('test.txt','r') as f:
    lines = f.readlines()
    mp = []
    for line in lines:
        mp.append(line.split(' ')[0])
    print (mp)