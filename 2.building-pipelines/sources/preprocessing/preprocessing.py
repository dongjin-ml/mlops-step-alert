import os
import sys
import argparse
#import subprocess
from distutils.dir_util import copy_tree
    
class preprocess():
    
    def __init__(self, args):
         
        self.args = args
        print (self.args)
       
    
    def logic(self, input_data_path):
        
        ## your logic here
        
        ## output(results) examples
        copy_tree(input_data_path, os.path.join(self.args.prefix_prep, "output", "train"))
        copy_tree(input_data_path, os.path.join(self.args.prefix_prep, "output", "validation"))
        copy_tree(input_data_path, os.path.join(self.args.prefix_prep, "output", "test"))
               
    def execution(self, ):
                
        input_data_path = os.path.join(self.args.prefix_prep, "input")
        print (f"input list: {input_data_path}: {os.listdir(input_data_path)}")
        self.logic(input_data_path)
        
if __name__ == "__main__":
        
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix_prep", type=str, default="/opt/ml/processing/")
    parser.add_argument("--region", type=str, default="ap-northeast-2")
    args, _ = parser.parse_known_args()
           
    print("Received arguments {}".format(args))
    os.environ['AWS_DEFAULT_REGION'] = args.region
    
    prep = preprocess(args)
    prep.execution()
    