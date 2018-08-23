import rospy
from simple_rl.tasks.FetchPOMDP import cstuff
from simple_rl.tasks.FetchPOMDP import RoboFetch
from std_msgs.msg import String

global human_feedback
global last_gesture
last_gesture = []
human_feedback = [None,None]  #[gesture,speech]

def gesture_callback(data):
    global last_gesture
    global human_feedback
    #print "gestures occuring!"
    #print "gesture!"
    #print data.data

    head_hand = data.data.split("^")
    gesture_list = []
    #print head_hand[1].split(" ")
    for i in head_hand[0].split(" "):
        #print i
        if i != '':
            gesture_list.append(float(i))
    for i in head_hand[1].split(" "):
        if i != '':
            gesture_list.append(float(i))
    
    if last_gesture != gesture_list:
        human_feedback[0] = gesture_list 
        last_gesture = gesture_list
    else:
        human_feedback[0] = None
    

def speech_callback(data):
    #print "speech occuring!"
    global human_feedback
    #print "speech!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1"
    #print data.data
    human_feedback[1] = data.data


rospy.init_node('fetch_node',anonymous=True)

rospy.Subscriber('human_gesture', String, gesture_callback)
rospy.Subscriber('google_speech', String, speech_callback)
right_arm = rospy.Publisher('/ein/right/forth_commands', String, queue_size=10)
left_arm = rospy.Publisher('/ein/left/forth_commands', String, queue_size=10)
eyes = rospy.Publisher('/baxter_eyes', String, queue_size=10)
rate = rospy.Rate(0.5)

height_offset = 0.2

#rospy.spin()
fetch_policy = RoboFetch()
while not rospy.is_shutdown():
    #print "hey!"
    #print human_feedback
    #DO WORK!
    # try:
    robot_action =  fetch_policy.act(human_feedback)
    print robot_action
    print fetch_policy.pomdp.cur_belief["desired_item"]
    human_feedback[1] = None #use the speech up
    if robot_action != "wait":
        (action_type, item_num) = robot_action.split(" ")
        if action_type == "pick":
           print "going to pick!"
           item_info = fetch_policy.pomdp.items[int(item_num)]
           print "going to pick2!"
           loc = item_info["location"]
           print "going to pick3!"
           msg = str(loc[0]) + " " + str(loc[1]) + " " + str(loc[2]+height_offset) + " 1 0 0 0 moveToEEPose"
           print "going to pick4!"
           right_arm.publish(msg)
           print "going to pick5!"
           # left_arm.publish("openGripper")
           # time.sleep(1)
           # left_arm.publish("closeGripper")
        if action_type == "point":
           item_info = fetch_policy.pomdp.items[int(item_num)]
           loc = item_info["location"]
           msg = str(loc[0]) + " " + str(loc[1]) + " " + str(loc[2]+height_offset) + " 1 0 0 0 moveToEEPose"
           right_arm.publish(msg)
        if action_type == "look":
           item_info = fetch_policy.pomdp.items[int(item_num)]
           loc = item_info["location"]
           msg = str(loc[0]) + " " + str(loc[1]) + " " + str(loc[2])
           eyes.publish(msg)
    # except Exception as e:
    #     print(e)
    rate.sleep()
