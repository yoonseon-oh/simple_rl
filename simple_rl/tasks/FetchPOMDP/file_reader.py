import os
import json

print(str(os.path.dirname(os.path.realpath(__file__))))
def load_json(file_name):
	# return {"items":["cup"]}
	directory = os.path.dirname(os.path.realpath(__file__))
	with open(directory + "/" + file_name) as json_data:
		return json.load(json_data)