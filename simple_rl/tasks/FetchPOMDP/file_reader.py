import os, sys
import json, pickle
import csv


# print(str(os.path.dirname(os.path.realpath(__file__))))
def load_json(file_name):
	# return {"items":["cup"]}
	directory = os.path.dirname(os.path.realpath(__file__))
	with open(directory + "/" + file_name) as json_data:
		return json.load(json_data)

def write_csv(data_as_list, file_name = None, directory =None):
	#TODO update to work with both Python 2 and 3
	if len(data_as_list) == 0 or type(data_as_list[0]) != list:
		num_dimensions = 1
	else:
		num_dimensions = 2
	if directory is None:
		directory = os.path.dirname(os.path.realpath(__file__))
	file_name = directory + "/" + file_name
	if file_name is not None:
		if sys.version[0] == '3':
			with open(file_name, 'w') as myfile:
				wr = csv.writer(myfile, quoting=csv.QUOTE_MINIMAL)
				#If data is one dimensional or empty
				if num_dimensions == 1:
					wr.writerow(data_as_list)
				else:
					#If data has multiple rows, write each
					for i in range(len(data_as_list)):
						wr.writerow(data_as_list[i])
		elif sys.vetsion[0] == "2":
			with open(file_name, 'wb') as myfile:
				wr = csv.writer(myfile, quoting=csv.QUOTE_MINIMAL)
				#If data is one dimensional or empty
				if num_dimensions == 1:
					wr.writerow(data_as_list)
				else:
					#If data has multiple rows, write each
					for i in range(len(data_as_list)):
						wr.writerow(data_as_list[i])
	if num_dimensions == 1:
		data_as_csv = ""
		for i in range(len(data_as_list)):
			if i != 0:
				data_as_csv += ","
			data_as_csv += str(data_as_list[i])
	else:
		data_as_csv = ""
		for i in range(len(data_as_list)):
			for j in range(len(data_as_list[i])):
				if j != 0:
					data_as_csv += ","
				data_as_csv += str(data_as_list[i][j])
			data_as_csv += "\n"
	return data_as_csv


def get_json_from_PBVI_pickle(file_name):
	directory = os.path.dirname(os.path.realpath(__file__)) + "/"
	file_name = directory + file_name
	output_name = file_name + ".json"
	p = pickle.load(open(file_name,"rb"))
	results = p["results"]
	results["pomdp"] = p["pomdp config"]
	with open(output_name, 'w') as fp:
		json.dump(results, fp, indent=4)
	return results