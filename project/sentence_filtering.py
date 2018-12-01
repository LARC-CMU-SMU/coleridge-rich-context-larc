from nltk.tokenize import sent_tokenize
import lsarray1
import aux_functions as aux_fun
auxfun=aux_fun.AuxFunClass
import os
import json

RESOURCE_PATH = 'resources/'
ABBS_PATH = RESOURCE_PATH + 'listofabb_dataset.txt'
PHRS_PATH = RESOURCE_PATH + 'listofPhrase_dataset.txt'

class SentenceFilterClass:


    # This function checks if a special feature exists in a context.
    def sep_checK(self,abb,cont):
            punc= lsarray1.Punctuation + lsarray1.Punctuation1 + lsarray1.Punctuation2 + [' '] + ['s']
            while abb in cont:
                    index=cont.find(abb)
                    lent=len(abb)
                    if cont[index+lent:index+lent+1] in punc or cont[index+lent:index+lent+1].isdigit():
                        if cont.find(abb)==0:
                            return True
                        elif cont[index-1:index] in punc or cont[index-1:index]==' ':
                            return True
                    elif index+lent==len(cont):
                        if cont.find(abb)==0:
                            return True
                        elif cont[index-1:index] in punc:
                            return True

                    try:
                        cont=cont[index+1:]
                    except:
                        return False
            return False


    #read papers
    def read_context(self, filename):
        f = open(filename, 'r', encoding = 'utf-8')

        content = ''
        for line in f:
            content = content + line.rstrip('\n')  + ' '
        return content[:-1]


    def final_approach(self, filename, ABBS_PATH = ABBS_PATH, PHRS_PATH = PHRS_PATH):
        context = self.read_context(filename)
        ls = sent_tokenize(context)

        abb1= auxfun.readtoarr2(auxfun,ABBS_PATH)
        abb2= auxfun.readtoarr2(auxfun, PHRS_PATH)

        candidate=[]
        for item in abb1:
            if self.sep_checK(item,context):
                candidate.append(item)
        for item in abb2:
            if self.sep_checK(item,context):
                candidate.append(item)

        #find whether there are exact macthes
        #return sentencesa and features
        textlist,abbinTlist= auxfun.sepfinder(auxfun, candidate, ls)

        abbinTlist1=list(set(abbinTlist))
        textlist1=list(set(textlist))

        lsallsplit=[]
        for itemabb_q in abbinTlist1:
           for itemsenasquery in textlist1:
                if self.sep_checK(itemabb_q,itemsenasquery):
                    neulistofquery= auxfun.querysplitter(auxfun, itemsenasquery, itemabb_q)
                    lsallsplit=lsallsplit+neulistofquery
        textlist1=list(set(lsallsplit))

        return textlist1


if __name__ == '__main__':
    ma = sentence_filter_class()
    textlist = ma.final_approach('test_data/files/text/143.txt')
    print(textlist)
