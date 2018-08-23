import math, random, json
import datetime as datetime
from std_msgs.msg import String
import rospy

eyes = rospy.Publisher('/baxter_eyes', String, queue_size=10)
rate = rospy.Rate(1)

output_directory = "./Precision tests logs/"


# geometry = ("circle",-math.pi/2,math.pi/2,0.5,0)
def cylindrical_to_cartesian_coordinates(angle, radius, height):
	x = radius * math.cos(angle)
	y = radius * math.sin(angle)
	z = height
	return [x, y, z]


def sample_from_line(start, end):
	length_squared = 0
	for i in range(len(start)):
		length_squared += (end[i] - start[i]) ** 2
	length = length_squared ** 0.5
	t = length * random.random()
	loc = []
	for i in range(len(start)):
		loc.append(start[i] + t * (end[i] - start[i]))
	return loc


def look_precision_test_circle(min_angle=-math.pi / 2, max_angle=math.pi / 2, radius=0.5, height=0, n=10):
	"""
	Looks at location randomly sampled from an arc of a circle parallel to the ground. Current item configs are in a line.
	:param min_angle:
	:param max_angle:
	:param radius:
	:param height:
	:param n:
	:return:
	"""
	# TODO test on baxter
	true_angles = []
	interpreted_angles = []
	num_samples = 0
	while num_samples < n and rospy.is_shutdown():
		# Scale up [0,1] to the size of the angle range and add the min_angle so that it corresponds to 0 and the max to 1
		angle = random.random() * (max_angle - min_angle) + min_angle
		print("Looking at angle: " + str(angle))
		true_angles.append(angle)
		loc = cylindrical_to_cartesian_coordinates(angle, radius, height)
		print("Looking at point: " + str(loc))
		msg = str(loc[0]) + " " + str(loc[1]) + " " + str(loc[2])
		eyes.publish(msg)
		rate.sleep()
	data = {"true_angles": true_angles, "min_angle": min_angle, "max_angle": max_angle, "radius": radius,
	        "height": height, "geometry": "circle"}
	logname = output_directory + "look precision test circle" + str(datetime.now()).replace(":", ".")[:22] + ".json"
	json.dump(data, open(logname, "w"))


def look_precision_test_line(start, end, n=10):
	"""
	Looks at location randomly sampled from an arc of a circle parallel to the ground. Current item configs are in a line.
	:param min_angle:
	:param max_angle:
	:param radius:
	:param height:
	:param n:
	:return:
	"""
	# TODO test on baxter
	true_locations = []
	interpreted_angles = []
	num_samples = 0
	while num_samples < n and rospy.is_shutdown():
		# Sample a point on the line
		loc = sample_from_line(start, end)
		true_locations.append(loc)
		print("Looking at point: " + str(loc))
		msg = str(loc[0]) + " " + str(loc[1]) + " " + str(loc[2])
		eyes.publish(msg)
		rate.sleep()
	data = {"true_locations": true_locations, "start": start, "end": end, "geometry": "line"}
	logname = output_directory + "look precision test line" + str(datetime.now()).replace(":", ".")[:22] + ".json"
	json.dump(data, open(logname, "w"))


look_precision_test_line(start=[0.5085, -0.66, -0.116], end=[0.36, 0.71, -0.12])
