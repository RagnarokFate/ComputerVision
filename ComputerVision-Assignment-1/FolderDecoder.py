import os
import numpy as np

# Extract Dimensions for the full picture from .txt file.
def get_height_width(filename):
    filename_array = filename.split("_")

    for i in range(0, len(filename_array)):
        if (filename_array[i] == "H"):
            height = filename_array[i + 1]
        if (filename_array[i] == "W"):
            width = filename_array[i + 1]
    return height, width

def Textfile(Puzzle_Folder_Path):
    files_in_folder = os.listdir(Puzzle_Folder_Path)
    for file in files_in_folder:
        if file.startswith("warp"):
            Textfile_Name = file
            TextFile_Matrix = np.loadtxt(Puzzle_Folder_Path + "/" + file)

    TextFile_Height, Textfile_Width = get_height_width(Textfile_Name)
    return Textfile_Name, TextFile_Matrix, TextFile_Height, Textfile_Width

# Puzzle_Folder_Path = MAIN PATH TO PUZZLE FOLDER
# Type = Type of Puzzle Transformation
# TextFile = "warp.....txt"
# TextFile_Height = height of the image that is written in txt file
# Textfile_Width = width of the image that is written in txt file
# Textfile_Matrix = The Matrix which is written in txt file
# puzzle_pieces = array of the pieces names
# Puzzle_Pieces_Number = length of the array above


def PuzzleLoader(Puzzle_Folder_Path_): # main folder
    Puzzle_Folder_Path = Puzzle_Folder_Path_
    Type = ""
    if (str(Puzzle_Folder_Path).startswith("puzzles/puzzle_affine")):
        Type = "affine"
    elif (str(Puzzle_Folder_Path).startswith("puzzles/puzzle_homography")):
        Type = "homography"

    Textfile_Name, TextFile_Matrix, TextFile_Height, Textfile_Width = Textfile(Puzzle_Folder_Path + "/")

    # Seek into pieces!
    pieces_paths = []
    Pieces_folder = Puzzle_Folder_Path_ + "/" + "pieces" + "/"
    puzzle_pieces = os.listdir(Pieces_folder)
    Puzzle_Pieces_Number = len(puzzle_pieces)

    for piece in puzzle_pieces:
        pieces_paths.append(Puzzle_Folder_Path + "/" + "pieces" + "/" + piece)

    return Puzzle_Folder_Path,Type,Textfile_Name,TextFile_Height, Textfile_Width,TextFile_Matrix,puzzle_pieces,Puzzle_Pieces_Number,pieces_paths
    #return Puzzle_Folder_Path,Type,puzzle_pieces,Puzzle_Pieces_Number,pieces_paths


def LoadFolders():
    Folders_Paths = []

    for i in range (1,11):
        Folders_Paths.append("puzzle_affine_" + str(i))

    for i in range (1,11):
        Folders_Paths.append("puzzle_homography_" + str(i))

    return Folders_Paths