# import json, os
# def load_json(file_name):
# 	#Doesn't work - __file__ not defined for c modules.
# 	current_file_full_path = os.path.realpath(__file__)
# 	directory = os.path.dirname(current_file_full_path)
# 	json_path = directory + "\\" + file_name
# 	print("json_path")
# 	print(json_path)
# 	with open(json_path) as json_data:
# 		return json.load(json_data)
__all__ = ['cstuff']
from .FetchPOMDPClass import FetchPOMDP
