# The file contains just some auxiliary functions.
import re
import nltk
import lsarray1

class AuxFunClass:

    
    def __init__(self):
        pass


    def readtoarr2(self,str):
        with open(str, "r", encoding="utf-8") as f:
             mylist = list(f)
        fl=[]
        for item in mylist:
            fl.append(item.rstrip('\n'))
        return  fl


    def querysplitter(self,query,abb_refrence):
        lssplits=[]

        query_proc = self.rmovepu(self,query)
        listofindexs=[m.start() for m in re.finditer(abb_refrence, query_proc)]
        lentofindex=len(listofindexs)

        for i in listofindexs:
            left_words = nltk.word_tokenize(query_proc[:i])
            right_words = nltk.word_tokenize(query_proc[i:])

            context = []
            if len(left_words) >= 5:
                context = context + left_words[-5:]
            else:
                context = context + left_words

            if len(right_words) >= 6:
                context = context + right_words[:6]
            else:
                context = context + right_words
            #print(len(context))
            lssplits.append(' '.join(context))

        return lssplits


    #find that the seperation of features and extracted words are same
    def sepfinder(self,abb,ls):
        punc= lsarray1.Punctuation + lsarray1.Punctuation1 + lsarray1.Punctuation2 + ["I"] + [" "]
        list1=[]
        list2=[]
        for item1 in abb:
            for item in ls:
                if item1 in item:
                    index=item.find(item1)
                    lent=len(item1)
                    if item[index+lent:index+lent+1] in punc or item[index+lent:index+lent+1].isdigit():
                        if item.find(item1)==0:
                            list1.append(item)
                            list2.append(item1)
                        elif item[index-1:index] in punc:
                            list1.append(item)
                            list2.append(item1)
                    elif index+lent==len(item):
                        if item.find(item1)==0:
                            list1.append(item)
                            list2.append(item1)
                        elif item[index-1:index] in punc:
                            list1.append(item)
                            list2.append(item1)
        return list1,list2


    def rmovepu(self,item):
        puli=lsarray1.Punctuation2+lsarray1.Punctuation1+lsarray1.Punctuation + ['(', ')']
        puli=list(set(puli))
        #puli.remove(')')
        fil_words = [word for word in nltk.word_tokenize(item) if word not in puli]
        return ' '.join(fil_words)
