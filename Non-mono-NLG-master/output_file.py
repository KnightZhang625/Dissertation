import sys
from classify import Classify

class Output(object):

    def __init__(self,type):

        if type == 'train':
            self.c = Classify(train=False,type='train')
        else:
            self.c = Classify(train=False,type='valid')
        self.c.load_data()
        self.labels = self.c.temp

        self.tgt_sentences = self.c.sentences
        self.src_sentences = self.c.raw_sentence

        self.trainData = []

    def write_file(self):
        for l,data in self.labels.items():
            self.labels[l] = data[0]
            temp_tgt_data,temp_src_data = self.seperate(l)
            self.write_tgt(temp_tgt_data,l)
            self.write_src(temp_src_data,l)

    def seperate(self,l):
        temp_tgt_data = [self.tgt_sentences[i] for i in self.labels[l]]
        temp_src_data = [self.src_sentences[i] for i in self.labels[l]]
        return temp_tgt_data,temp_src_data

    def write_tgt(self,data,l):
        if self.c.type == 'train':
            f = open('cache/data/1_tgt_label_%s.txt'%(l),'w')

            for sentence in data:
                f.write(sentence+'\n')
                f.flush()

            f.close()
        else:
            f = open('cache/data/2_tgt_label_%s.txt'%(l),'w')

            for sentence in data:
                f.write(sentence+'\n')
                f.flush()

            f.close()

    def write_src(self,data,l):
        if self.c.type == 'train':
            f = open('cache/data/1_src_label_%s.txt'%(l),'w')

            for sentence in data:
                f.write(sentence+'\n')
                f.flush()

            f.close()
        else:
            f = open('cache/data/2_src_label_%s.txt'%(l),'w')

            for sentence in data:
                f.write(sentence+'\n')
                f.flush()

            f.close()

if __name__ == '__main__':

    o = Output(sys.argv[1])
    o.write_file()








