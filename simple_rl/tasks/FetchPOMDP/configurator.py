import json
import random

old_items = [{"shape": "spoon", "color": "red", "location": [100, 0, 0]},
             {"shape": "cup", "color": "blue", "location": [0, 100, 0]},
             {"shape": "cup", "color": "red", "location": [0, 0, 100]},
             {"shape": "fork", "color": "red", "location": [50, 50, 0]},
             {"shape": "marker", "color": "black", "location": [50, 50, 50]}]


def load_json(file_name):
	with open(file_name) as json_data:
		return json.load(json_data)


# def raw_config_to_dict(file_name):
# 	with open(file_name) as f:
# 		lines = f.read().splitlines()
# 		converted_lines = []
# 	for line in lines:
# 		if line[0] != "#":
# 			terms = line.split(" = ")
# 			new_line = "\"" + terms[0] + "\":" + terms[1]
# 			converted_lines.append(new_line)

def generate_location(max=100):
	location = [int(random.random() * max), int(random.random() * max), int(random.random() * max)]
	while location[0] == location[1] == location[2] == 0:
		location = [int(random.random() * max), int(random.random() * max), int(random.random() * max)]
	return location


def generate_location2(a, b, c):
	location = [int(random.random() * a), int(random.random() * b), int(random.random() * c)]
	while location[0] == location[1] == location[2] == 0:
		location = [int(random.random() * a), int(random.random() * b), int(random.random() * b)]
	return location


def sample(a):
	return a[random.randrange(len(a))]


default_shapes = ["marker", "cup", "spoon", "fork", "ball", "cylinder"]
default_colors = ["red", "green", "blue", "yellow", "white","purple"]


def generate_items(shapes=default_shapes, colors=default_colors, n=6):
	items = [{"shape": sample(shapes), "color": sample(colors), "location": generate_location()} for i in range(n)]
	for i in range(n):
		print(str(items[i]) + ",")
	return items


def generate_items2(shapes=default_shapes, colors=default_colors, n=6):
	items = [{"shape": sample(shapes), "color": sample(colors), "location": generate_location(max=2)} for i in range(n)]
	for i in range(n):
		print(str(items[i]) + ",")
	return items


def generate_items3(shapes=default_shapes, colors=default_colors, n=6):
	items = [{"shape": sample(shapes), "color": sample(colors), "location": generate_location2(100, 0, 0)} for i in
	         range(n)]
	for i in range(n):
		print(str(items[i]) + ",")
	return items


def generate_items_unambiguous(shapes=default_shapes, colors=default_colors, n=6):
	items = []
	for i in range(n):
		# location = [(i+1) * 1000.0/n, 0, 0]
		location = generate_location(max=100)
		items.append({"shape": shapes[i], "color": colors[i], "location": location})
	return items


def union_dictionary(dictionary):
	un = set()
	for value in dictionary.values():
		un.update(value)
	return list(un)


def get_relevant_words(item_index, bag, items):
	words = set()
	item = items[item_index]  # parameter is int, we want the dict
	for att in ATTRIBUTES:
		words.update(bag[item[att]])
	return list(words)


def get_irrelevant_words(item_index, bag,items):
	item = items[item_index]  # fed int, want dict
	words = set()
	keys = set(bag.keys())
	for att in ATTRIBUTES:
		keys.remove(item[att])
	for key in keys:
		words.update(bag[key])
	return list(words)


ATTRIBUTES = ["shape", "color"]
calibrated_items = [
    {
      "shape": "cylinder",
      "color": "blue",
      "location": [
        0.5085,
        -0.66,
        -0.116
      ]
    },
    {
      "shape": "ball",
      "color": "red",
      "location": [
        0.48,
        -0.398,
        -0.118
      ]
    },
    {
      "shape": "marker",
      "color": "purple",
      "location": [
        0.469,
        -0.152,
        -0.118
      ]
    },
    {
      "shape": "ball",
      "color": "blue",
      "location": [
        0.43,
        0.189,
        -0.12
      ]
    },
    {
      "shape": "block",
      "color": "yellow",
      "location": [
        0.405,
        0.423,
        -0.118
      ]
    },
    {
      "shape": "marker",
      "color": "green",
      "location": [
        0.36,
        0.71,
        -0.12
      ]
    }
  ]
calibrated_bag_of_words = {
    "spoon": [
      "spoon",
      "dipper"
    ],
    "cup": [
      "cup",
      "mug"
    ],
    "fork": [
      "fork"
    ],
    "block": [
      "block"
    ],
    "cylinder": [
      "cylinder"
    ],
    "red": [
      "red",
      "pink",
      "vermillion"
    ],
    "blue": [
      "blue",
      "turquoise"
    ],
    "yellow": [
      "yellow"
    ],
    "green": [
      "green"
    ],
    "purple": [
      "purple"
    ],
    "marker": [
      "marker",
      "expo"
    ],
    "ball": [
      "ball",
      "sphere"
    ],
    "phone": [
      "phone"
    ],
    "white": [
      "white"
    ],
    "position": []
  }
items = generate_items(n=10)
# bag_of_words = {
# 	"spoon": ["spoon", "dipper"],
# 	"cup": ["cup", "mug"],
# 	"fork": ["fork"],
# 	"red": ["red", "pink", "vermillion"],
# 	"blue": ["blue", "turquoise"],
# 	"yellow": ["yellow"],
# 	"green": ["green"],
# 	"black": ["black", "dark"],
# 	"marker": ["marker", "expo"],
# 	"ball": ["ball", "sphere"],
# 	"phone": ["phone"],
# 	"white": ["white"],
# 	"position": []
# }
bag_of_words = calibrated_bag_of_words
items = calibrated_items
all_words = union_dictionary(bag_of_words)
relevant_words = [get_relevant_words(i, bag_of_words, items) for i in range(len(items))]
irrelevant_words = [get_irrelevant_words(i, bag_of_words, items) for i in range(len(items))]
config = {"items":  items,
          "desired_item": 0,
          "num_items": len(items),
          "bag_of_words": bag_of_words,
          "attributes": ATTRIBUTES,
          "positive_responses":["yes", "yeah", "sure", "yup"],
          "negative_responses":["no", "nope", "other", "not"],
          "p_g": 0.1,
          "p_l": 0.95,
          "p_r_match": 0.99,
          "alpha": 0.1,
          "std_theta": 0.15,
          "std_theta_look": 0.3,
          "point_cost": -6,
          "look_cost": -0.333,
          "wait_cost": -0.167,
          "wrong_pick_cost": -20,
          "correct_pick_reward": 10,
          "gamma": 0.9}

with open('config.json', 'w') as fp:
	json.dump(config, fp)
print(" ")
print(config.keys())
