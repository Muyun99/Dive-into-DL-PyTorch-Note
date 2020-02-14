import torchvision
import os
import urllib 
import nltk
import zipfile

def check():
    flag = True
    if(os.path.exists("./datasets") == False):
        flag = False
    return flag

def download():
    # create datasets folder
    if(os.path.exists("./datasets") == False):
        os.mkdir("./datasets")
    
    # 1. download FashionMNIST DataSets
    mnist_train = torchvision.datasets.FashionMNIST(root='./datasets/FashionMNIST', train=True, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(root='./datasets/FashionMNIST', train=False, download=False)

    # 2. download timemachine.txt
    url = "http://www.gutenberg.org/files/35/35-0.txt"
    urllib.request.urlretrieve(url, "./datasets/timemachine.txt")

    # 3. download jaychou_lyrics.txt
    url = "https://raw.githubusercontent.com/d2l-ai/d2l-zh/master/data/jaychou_lyrics.txt.zip"
    urllib.request.urlretrieve(url, "./datasets/jaychou_lyrics.zip")

    path = "./datasets/jaychou_lyrics/"
    if(os.path.exists(path) == False):
        os.mkdir(path)
    zip_file = zipfile.ZipFile("./datasets/jaychou_lyrics.zip")
    for names in zip_file.namelist():
        zip_file.extract(names,path)
    zip_file.close()
    os.remove("./datasets/jaychou_lyrics.zip")

    # 4. download d2lzh
    url = "https://muyun-blog-pic.oss-cn-shanghai.aliyuncs.com/d2lzh"
    urllib.request.urlretrieve(url, "./utils/d2lzh.zip")

    
    path = "./utils/"
    if(os.path.exists(path) == False):
        os.mkdir(path)
    zip_file = zipfile.ZipFile("./utils/d2lzh.zip")
    for names in zip_file.namelist():
        zip_file.extract(names,path)
    zip_file.close()
    os.remove("./utils/d2lzh.zip")
    


if __name__ == "__main__":
    # if(check() == False)
    #     download()
    download()