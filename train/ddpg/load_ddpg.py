import csv
import threading
import os
import gym
import gym_autonmscar
from stable_baselines import DDPG

model_name = "/ddpg-models2/ddpg-model_567000.pkl"
fps = 0
seconds = 0

def get_fps():
    global fps, seconds, writer
    print("step per second: ", fps)
    if seconds == 0:
        writer.writerow(["seconds", "fps"])
    else:
        writer.writerow([seconds, fps])
    fps = 0
    seconds += 1
    threading.Timer(1, get_fps).start()


if __name__ == "__main__":
    env = gym.make('autonmscarContinuous-v0')
    model = DDPG.load(os.path.dirname(
        os.path.realpath(__file__)) + model_name)

    output_file = open('fps_ddpg_play.csv', 'w', encoding='utf-8')
    global writer
    writer = csv.writer(output_file)
    fps_thread = threading.Thread(target=get_fps)
    fps_thread.start()

    obs = env.reset()
    done = False
    for i in range(1000):
        if done:
            env.reset()
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        env.render()
        fps += 1
    
    output_file.close()
