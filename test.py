import os
import json 
from collections import OrderedDict   
    
json_content = {
    'images':OrderedDict(),
    'annotation':OrderedDict()
}

json_content['images']['new'] = 1
with open('test.json', 'r') as f:
    a=  json.load(f)
print(a['images']['new'])