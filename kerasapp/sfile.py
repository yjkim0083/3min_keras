import datetime
import uuid
import os

def unique_filename(type='uuid'):
    if type == 'datetime':
        filename = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
    else :
        filename = str(uuid.uuid4())
    return filename

def make_new_fold(prefix="output_", type="datetime"):
    suffix = unique_filename('datetime')
    foldname = 'output_' + suffix
    os.makedirs(foldname)
    return foldname