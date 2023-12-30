import os;
import numpy as np

def Extract_MaxDisp(File_Name):
    with open(File_Name, "r") as file:
        # Read the entire content of the file
        file_content = file.read()

    return file_content

def Extract_K(File_Name):
    matrix = np.loadtxt(fname=File_Name)


    return matrix

def Extract_TestFiles(FolderName):
    print("Currently : " + FolderName)
    files = os.listdir(FolderName)
    pictures = []
    for file in files:
        if(file.endswith(".jpg")):
            pictures.append(file)
        if(file.startswith("K")):
            K_Value = Extract_K(FolderName + "/" + file)
        if (file.startswith("max_disp")):
            MaxDisp_Value = Extract_MaxDisp(FolderName + "/" + file)

    return pictures,K_Value,MaxDisp_Value;



