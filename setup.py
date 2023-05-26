import os 
os.makedirs('./datasets', exist_ok = True) 
os.makedirs('./trained_models', exist_ok = True)  


os.chdir('./datasets') 
os.system("git clone https://github.com/sahoonihar/ToxicBias_CoNLL_2022.git")
os.system("git clone https://github.com/marcoguerini/CONAN.git") 
os.chdir('../')


