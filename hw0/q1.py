import sys



def makeSelectedList(col):
    
    selectedList=[]
    
    for line in open(sys.argv[2],'r'):
        numCount=-2
        numP1=1
        for idx,ch in enumerate(line):
            if ch==' ' or ch=='\n':
                numCount+=1
                if numCount==col :
                    num=float(line[numP1:idx])
                    selectedList.append(num)     
                    break
                else:
                    numP1=idx+1

    return sorted(selectedList)

try:
    col = int(sys.argv[1])
    if(col<0 or col>10):
        raise IndexError('not a correct colum')
    
    try:
        
        wData = ','.join(str(n) for n in makeSelectedList(col))
        with open("ans1.txt",'w') as file:
            file.write(wData)
    except IndexError:
        print('plase provide file name')
    except FileNotFoundError:
        print('cannnot find {0}'.format(sys.argv[2]))
except IndexError as e:
    print(type(e),str(e))
    raise e
