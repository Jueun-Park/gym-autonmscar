import csv
import threading
import os
import gym
import gym_autonmscar
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import DQN

model_name = "/dqn-models/dqn-model_544000.pkl"
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
    env = gym.make('autonmscar-v0')
    env = DummyVecEnv([lambda: env])
    model = DQN.load(os.path.dirname(
        os.path.realpath(__file__)) + model_name)

    output_file = open('fps_dqn_play.csv', 'w', encoding='utf-8')
    global writer
    writer = csv.writer(output_file)
    fps_thread = threading.Thread(target=get_fps)
    fps_thread.start()

    obs = env.reset()
    for i in range(1000):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        env.render()
        fps += 1

    output_file.close()
