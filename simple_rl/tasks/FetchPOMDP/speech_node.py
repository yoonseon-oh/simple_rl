# NOTE: this example requires PyAudio because it uses the Microphone class

import speech_recognition as sr
import rospy
from std_msgs.msg import String

# obtain audio from the microphone
r = sr.Recognizer()

pub = rospy.Publisher('google_speech', String, queue_size=10)
rospy.init_node('fetch_speech', anonymous=True)
rate = rospy.Rate(10)

try:
    while True:
        with sr.Microphone() as source:
            r.adjust_for_ambient_noise(source)
            print("Say something!")
            audio = r.listen(source)
 
        # recognize speech using Google Speech Recognition
        try:
            text = r.recognize_google(audio)
            # for testing purposes, we're just using the default API key
            # to use another API key, use `r.recognize_google(audio, key="GOOGLE_SPEECH_RECOGNITION_API_KEY")`
            # instead of `r.recognize_google(audio)`
            print("Google Speech Recognition thinks you said " + text)
            pub.publish(text)
        
        except sr.UnknownValueError:
            print("Google Speech Recognition could not understand audio")
        except KeyboardInterrupt:
            print ("interrupted again!")
        except sr.RequestError as e:
            print("Could not request results from Google Speech Recognition service; {0}".format(e))
except KeyboardInterrupt:
    print "interrupted"


