import argparse 
from pathlib import Path  
import os 
import pandas as pd  
import re, string
from sklearn.preprocessing import MultiLabelBinarizer


parser = argparse.ArgumentParser() 
parser.add_argument("-p1","--path1",default=None,help="path for dataset folder of ToxicBias_CoNLL_2022 dataset") 
parser.add_argument("-p2","--path2",default=None,help="path for dataset folder that contains CONAN.csv file") 
parser.add_argument("-s","--save_path",default="./datasets/", help="folder path for csv files to save the augmented dataset and processed dataset")

args = parser.parse_args()  
args = vars(args)

class DataPrep(): 
    def __init__(self, path1, path2, save_path, df = None) -> None:
        self.path1 = path1 
        self.path2 = path2 
        self.save_path = save_path 
        self.combined_df = df 
        self.CLASSES = ['religion','race','gender','political','lgbtq','none']

        if self.path1 is not None: 
            train = pd.read_csv(os.path.join(self.path1, "toxicbias_train.csv")) 
            test = pd.read_csv(os.path.join(self.path1, "toxicbias_test.csv")) 
            val = pd.read_csv(os.path.join(self.path1, "toxicbias_val.csv")) 

            self.combined_df = pd.concat([train, test, val]).reset_index(drop = True)[['comment_text','bias','category']]

        if self.path2 is not None: 
            conan_df = pd.read_csv(os.path.join(self.path2, "CONAN/CONAN.csv")) 
            conan_df = conan_df[conan_df.cn_id.apply(lambda x : x.startswith('EN') or x.endswith('T1'))]
            conan_df[conan_df.hsSubType.apply(lambda x : 'economics' in x)]

            multi_conan_df = pd.read_csv(os.path.join(self.path2, "Multitarget-CONAN/Multitarget-CONAN.csv")) 
            multi_conan_df = multi_conan_df[(multi_conan_df.TARGET != 'other') & (multi_conan_df.TARGET != 'DISABLED')]


            new_MC_df = []
            for index, row in multi_conan_df.iterrows():
                # Call the custom function on the row and get the new rows
                row1,row2 = self.multi_conan_to_toxic_bias(row)
                
                # Append the new rows to the new DataFrame
                new_MC_df.append(row1)
                new_MC_df.append(row2)
            new_MC_df = pd.DataFrame(new_MC_df)

            new_conan_df = []
            for index, row in conan_df.iterrows():
                # Call the custom function on the row and get the new rows
                row1,row2 = self.conan_to_toxic_bias(row)
                
                # Append the new rows to the new DataFrame
                new_conan_df.append(row1)
                new_conan_df.append(row2)
            new_conan_df = pd.DataFrame(new_conan_df)

            final_df = pd.concat([self.combined_df,new_conan_df,new_MC_df])
            self.combined_df = final_df.reset_index(drop=True) 

    
    def save_file(self, filename): 
        print("Saving file to ", os.path.join(self.save_path,filename))
        self.combined_df.to_csv(os.path.join(self.save_path,filename),index=False) 


    # seg_categories = lambda x: set([s.strip() for s in x.split(',')])
    # encode_bias = lambda x: 1 if x == 'bias' else 0

    # Function to apply the transformations
    def apply_transformations(self): 
        # Apply seg_categories function to 'categories' column
        self.combined_df['category'] = self.combined_df['category'].apply(lambda x: set([s.strip() for s in x.split(',')]))
        # Apply encode_bias function to 'bias' column
        self.combined_df['bias'] = self.combined_df['bias'].apply(lambda x: 1 if x == 'bias' else 0)

        mlb = MultiLabelBinarizer(classes = self.CLASSES)
        encoded_categories = mlb.fit_transform(self.combined_df.category)
        encoded_df = pd.DataFrame(encoded_categories, columns=self.CLASSES)
        transformed_data = pd.concat([self.combined_df['comment_text'], encoded_df, self.combined_df['bias']], axis=1) 
        transformed_data = transformed_data[['comment_text', 'bias', 'religion', 'race', 'gender','political', 'lgbtq', 'none']]
        self.combined_df = transformed_data 
        print("Multi-Label One Hot Encoding Complete")

    

    def multi_conan_to_toxic_bias(self, row):
        hs = row.HATE_SPEECH
        cn = row.COUNTER_NARRATIVE
        target = ''
        if row.TARGET in ['JEWS','MUSLIMS']:
            target = 'religion'
        elif row.TARGET in ['WOMEN']:
            target = 'gender'
        elif row.TARGET in ['MIGRANTS']:
            target = 'political'
        elif row.TARGET in ['POC']:
            target = 'race'
        elif row.TARGET in ['LGBT+']:
            target = 'lgbtq'
        d1 = {'comment_text':hs,'bias':'bias','category':target}
        d2 = {'comment_text':cn,'bias':'neutral','category':'none'}
        return [d1,d2]


    def conan_to_toxic_bias(self, row):
        hs = row.hateSpeech
        cn = row.counterSpeech
        target = 'religion'
        categories = row.hsSubType.split(',')
        for i in range(len(categories)):
            categories[i] = categories[i].strip().lower()
        if set(categories) & set(['women','rapism']):
            target += ',gender'
        d1 = {'comment_text':hs,'bias':'bias','category':target}
        d2 = {'comment_text':cn,'bias':'neutral','category':'none'}
        return [d1,d2]
    


if __name__ == "__main__":  
    os.makedirs(args["save_path"], exist_ok=True)

    worker = DataPrep(args["path1"], args["path2"], args["save_path"]) 
    worker.apply_transformations() 
    worker.save_file("processed_df.csv")
     
