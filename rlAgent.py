import gymnasium as gym
import requests
import numpy as np
from stable_baselines3 import PPO
from os import makedirs
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import SubprocVecEnv

import torch as th


class ai2cyberEnv(gym.Env):
    def __init__(self, seqLength: int = 5, maxSteps: int = 20, verbose: bool = False, training: bool = False):
        super().__init__()
        self.seqLength: int = seqLength
        self.maxSteps: int = maxSteps
        self.verbose: bool = verbose
        self.baseUrl: str = "http://63.176.107.188:5005"
        self.uuid: str = None
        self.training: bool = training
        self.previousObservation: np.ndarray | None = None

        # The next two I could programatically get from the API, but for now I'm hardcoding them based on the API documentation and testing.
        self.action_space: gym.spaces.Discrete = gym.spaces.Discrete(3)
        self.observation_space: gym.spaces.Box = gym.spaces.Box(
            low=0.0, 
            high=1.0, 
            shape=(2 * self.seqLength,), 
            dtype=np.float32
        )
        self._initializeGame()


    def _initializeGame(self):
        payload = {"max_steps": self.maxSteps, "seq_length": self.seqLength}
        
        endpoint = f"{self.baseUrl}/new_game"
        response = requests.post(endpoint, headers={"Content-Type": "application/json"}, json=payload)
        
        if response.status_code == 200:
            data: dict = response.json()
            self.uuid = data["uuid"]
            if self.verbose:
                print(f"\nNew game UUID: {self.uuid}", flush=True)
            if not self.uuid:
                raise ValueError("API did not return a valid UUID.")
        else:
            raise ConnectionError(f"Failed to start new game. Status code: {response.status_code}, Response: {response.text}")


    def reset(self, seed=None, options=None, uuid: str = None):
        super().reset(seed=seed)
        if not uuid:
            self._initializeGame()
        endpoint = f"{self.baseUrl}/reset"
        response = requests.post(endpoint, headers={"Content-Type": "application/json"}, json={"uuid": self.uuid})
        if response.status_code == 200:
            data = response.json()
            observation = np.array(data.get("observation", []), dtype=np.float32)
            info = data.get("info", {})
            infoDict = {}
            if "goal_reached" in info:
                infoDict["goal_reached"] = True
            # Maybe I shuold return this instead of truncated?
            if "timeout_reset" in info:
                infoDict["timeout_reset"] = True
            return observation, infoDict

        else:
            raise ConnectionError(f"Failed to reset game. Status code: {response.status_code}, Response: {response.text}")
            
    def step(self, action: int):
        endpoint = f"{self.baseUrl}/step"
        response = requests.post(endpoint, headers={"Content-Type": "application/json"}, json={"uuid": self.uuid, "action": int(action)})
        if response.status_code == 200:
            data = response.json()
            observation = np.array(data.get("observation", []), dtype=np.float32)
            reward = data.get("reward", 0.0)
            done = data.get("done", False)
            info = data.get("info", {})
            infoDict = {}
            if "goal_reached" in info:
                infoDict["goal_reached"] = True
                done = True
            # This does not need to exist to be frank as it is never really used that way by the api.
            truncated = data.get("truncated", False)
            if "timeout_reset" in info:
                infoDict["timeout_reset"] = True
                truncated = True

            if self.verbose:
                print(f"Observation: {observation}")

            # if self.training and self.previousObservation is not None:
            #     # print(f"Action taken: {action}, Reward received: {reward}, Done: {done}, Truncated: {truncated}, Info: {infoDict}")
            #     ashesCur = observation[:self.seqLength]
            #     ashesPre = self.previousObservation[:self.seqLength]
            #     if np.sum(ashesCur) < np.sum(ashesPre):
            #         reward = -2.0
            if self.training and reward < 0:
                reward = reward * 10.0
            if self.training and reward > 0:
                reward = reward * 1.1

            # print(str(action) + "\t" + str(reward))

            return observation, float(reward), done, truncated, infoDict
        else:
            raise ConnectionError(f"Failed to take step. Status code: {response.status_code}, Response: {response.text}")
        
    # def 



class trainAgent:
    def __init__(self, 
                 lengths: list[int], 
                 logDir: str = "./out/logs",
                 modelDir: str = "./out/models",
                 evalLogDir: str = "./out/eval",
                 ):
        self.lengths = lengths
        self.logDir = logDir
        self.modelDir = modelDir
        self.evalLogDir = evalLogDir
        # self.learning_rate = 0.1

        # policy_kwargs = dict(activation_fn=th.nn.ReLU,net_arch=dict(pi=[64, 64], vf=[64, 64]),)

        self.hyperparams = {
            "learning_rate": 0.0005,
            "n_steps": 2048,
            "batch_size": 64,
            "n_epochs": 5,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "ent_coef": 0.05,
            "clip_range": 0.2,
        }

    # @staticmethod
    def makeEnv(seqLength: int, maxSteps: int):
        env = ai2cyberEnv(seqLength=seqLength, maxSteps=maxSteps)
        return env
        # return ai2cyberEnv(seqLength=seqLength, maxSteps=maxSteps)
    
    def trainPPO(self, seqLength: int, timeSteps: int):
        makedirs(self.logDir+f"/seq{seqLength}", exist_ok=True)
        makedirs(self.modelDir+f"/seq{seqLength}", exist_ok=True)
        makedirs(self.evalLogDir+f"/seq{seqLength}", exist_ok=True)

        maxSteps = seqLength * 4

        env = make_vec_env(lambda: ai2cyberEnv(seqLength=seqLength, maxSteps=maxSteps, training=True), n_envs=4, vec_env_cls=SubprocVecEnv)
        evalEnv = make_vec_env(lambda: ai2cyberEnv(seqLength=seqLength, maxSteps=maxSteps, training=True), n_envs=4, vec_env_cls=SubprocVecEnv)

        # TODO check the paths
        evalCallback = EvalCallback(evalEnv, best_model_save_path=self.evalLogDir+f"/seq{seqLength}", log_path=self.evalLogDir+f"/seq{seqLength}", eval_freq=5000, deterministic=True, render=False)

        model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=self.logDir+f"/seq{seqLength}", **self.hyperparams, policy_kwargs=dict(activation_fn=th.nn.ReLU,net_arch=dict(pi=[64, 64], vf=[64, 64]),))

        model.learn(total_timesteps=timeSteps, callback=evalCallback, progress_bar=True)
        model.save(self.modelDir+f"/ppo_ai2cyber_seq{seqLength}")

        print("Training complete.")



    def runAll(self):
        for length in self.lengths:
            timeSteps = 50000 + (length - 5) * 50000
            self.trainPPO(seqLength=length, timeSteps=timeSteps)


if __name__ == "__main__":
    trainer = trainAgent(lengths=[5, 6, 7])
    trainer.runAll()
