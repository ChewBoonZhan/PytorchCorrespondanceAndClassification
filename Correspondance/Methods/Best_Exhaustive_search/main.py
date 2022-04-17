import sys
import os

sys.path.insert(0, os.getcwd() + '\..\..\..\General_Helper_Function')
# example of import function from another folder

# import the function itself
import trial

if __name__ == '__main__':
    # called when runned from command prompt
    print(trial.trialFunction())
    print(trial.trialFunction2(10))