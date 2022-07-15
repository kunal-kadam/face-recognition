# import vlc
# p = vlc.MediaPlayer("filename.mp3")
# p.play()

import playsound, random
def play(topic="sleepy"):
    playsound.playsound(f'C:/Users/Kunal Kadam/Desktop/Intel/data/recordings/{topic}/filename{random.choice([0, 1, 2])}.mp3')

# play()