import gymnasium as gym
from collections import defaultdict
import numpy as np
import requests
# import json

class ai2cyberEnv(gym.Env):
    def __init__(self, seqLength: int = 5, maxSteps: int = 20):
        super().__init__()
        self.seqLength: int = seqLength
        self.maxSteps: int = maxSteps
        self.baseUrl: str = "http://63.176.107.188:5005"
        self.uuid: str = None
        self.action_space = gym.spaces.Discrete(3)  # 0: do nothing, 1: flip bit, 2: reset game. We get this from a custom post to the new_game api endpoint

        observation_space: gym.spaces.Box = gym.spaces.Box(
            # low=np.array([0.0, 0.0, 0.0]),  # min values for each element in the observation
            low=0.0,
            high=1.0,
            # high=np.array([self.seqLength - 1, self.maxSteps - 1, 1]),  # max values for each element
            dtype=np.float32
        )


    def new_game(self, seqLength: int = None, maxSteps: int = None):
        payload = {"max_steps": self.maxSteps, "seq_length": self.seqLength}
        endpoint = f"{self.baseUrl}/new_game"
        response = requests.post(endpoint, headers={"Content-Type": "application/json"}, json=payload)

        if response.status_code == 200:
            data: dict = response.json()
            self.uuid = data["uuid"]
            print(self.uuid)
            if not self.uuid:
                raise ValueError("API did not return a valid UUID.")
        else:
            raise ConnectionError(f"Failed to start new game. Status code: {response.status_code}, Response: {response.text}")
        
    
    def reset(self, uuid: str = None):
        super().reset()
        if not self.uuid:
            self.new_game()
        endpoint = f"{self.baseUrl}/reset"
        response = requests.post(endpoint, headers={"Content-Type": "application/json"}, json={"uuid": self.uuid})
        if response.status_code == 200:
            data = response.json()
            observation = np.array(data.get("observation", []), dtype=np.float32)
            info = data.get("info", {})
            infoDict = {}
            if "goal_reached" in info:
                infoDict["goal_reached"] = True
            if "timeout_reset" in info:
                infoDict["timeout_reset"] = True
            return observation, infoDict

        else:
            raise ConnectionError(f"Failed to reset game. Status code: {response.status_code}, Response: {response.text}")
        

        


class trainAgent:
    def __init__(self, 
                 base_url: str,
                 env: gym.Env,
                 learning_rate: float,
                 initial_epsilon: float,
                 epsilon_decay: float,
                 final_epsilon: float,
                 discount_factor: float = 0.95,
                ):
        """
        Args:
            env: The training environment
            learning_rate: How quickly to update Q-values (0-1)
            initial_epsilon: Starting exploration rate (usually 1.0)
            epsilon_decay: How much to reduce epsilon each episode
            final_epsilon: Minimum exploration rate (usually 0.1)
            discount_factor: How much to value future rewards (0-1)
        """
        self.env = env
        self.q_values = defaultdict(lambda: np.zeros(env.action_space.n))
        self.lr = learning_rate
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon
        self.discount_factor = discount_factor
        self.training_error = []


    def get_action(self, obs: tuple[int, int, bool]) -> int:
        """Choose an action using epsilon-greedy strategy.

        Returns:
            action: 0 (do nothing) or 1 (flip bit)
        """
        # With probability epsilon: explore (random action)
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()

        # With probability (1-epsilon): exploit (best known action)
        else:
            return int(np.argmax(self.q_values[obs]))


    def update(
        self,
        obs: tuple[int, int, bool],
        action: int,
        reward: float,
        terminated: bool,
        next_obs: tuple[int, int, bool],
    ):
        """Update Q-value based on experience.

        This is the heart of Q-learning: learn from (state, action, reward, next_state)
        """
        # What's the best we could do from the next state?
        # (Zero if episode terminated - no future rewards possible)
        future_q_value = (not terminated) * np.max(self.q_values[next_obs])

        # What should the Q-value be? (Bellman equation)
        target = reward + self.discount_factor * future_q_value

        # How wrong was our current estimate?
        temporal_difference = target - self.q_values[obs][action]

        # Update our estimate in the direction of the error
        # Learning rate controls how big steps we take
        self.q_values[obs][action] = (
            self.q_values[obs][action] + self.lr * temporal_difference
        )

        # Track learning progress (useful for debugging)
        self.training_error.append(temporal_difference)

    def decay_epsilon(self):
        """Reduce exploration rate after each episode."""
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)



    # def select_action(self, observation):
    #     # Always select the action corresponding to Ashe (assuming it's action 0)
    #     return 0


if __name__ == "__main__":
    env = ai2cyberEnv()
    env.new_game(5, 20)
    print(env.reset())