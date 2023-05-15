import zipfile
import os
import argparse

parser = argparse.ArgumentParser(prog='unzipy.py', description='this file extract file and save it in specific location')

parser.add_argument('--filename', '-fn', type=str, default=None)
parser.add_argument('--savepath', '-sp', type=str, default='./')

args = parser.parse_args()



with zipfile.ZipFile(os.path.join(args.savepath, args.filename), 'r') as zip_ref:
    zip_ref.extractall(args.savepath)