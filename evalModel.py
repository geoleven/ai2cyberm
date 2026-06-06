from os import path
from stable_baselines3 import PPO
from rlAgent import ai2cyberEnv
from numpy import mean, std
from json import dump

class evalModel:

    @staticmethod
    def writeResults(resultsFile: str, results: dict | list):
        with open(resultsFile, "w") as f:
            dump(results, f)

    @staticmethod
    def evaluate(modelPath: str, numEpisodes: int = 10, seqLength: int = 5, maxSteps: int | None = None, verbose: bool = False):

        if not path.exists(modelPath):
            raise ValueError("Model path does not exist")
        if not path.isfile(modelPath):
            modelPath = path.join(modelPath, f"ppo_ai2cyber_seq{seqLength}.zip")
        if maxSteps is None:
            maxSteps = seqLength * 4

        model = PPO.load(modelPath)
        env = ai2cyberEnv(seqLength=seqLength, maxSteps=maxSteps, verbose=verbose)

        rewards = []
        steps = []
        for episode in range(numEpisodes):
            obs, info = env.reset()
            done = False
            truncated = False
            episodeReward = 0.0
            episodeSteps = 0
            while not (done or truncated):
                action, _states = model.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = env.step(action)
                episodeReward += reward
                episodeSteps += 1

            print(f"Episode {episode + 1}: Reward = {episodeReward}")

            rewards.append(episodeReward)
            steps.append(episodeSteps)
            
        averageReward = mean(rewards)
        print(f"Average Reward over {numEpisodes} episodes: {averageReward}")
        stdReward = std(rewards)
        print(f"Reward Std Dev: {stdReward}")
        averageSteps = mean(steps)
        print(f"Average Steps: {averageSteps}")
        stdSteps = std(steps)
        print(f"Steps Std Dev: {stdSteps}")

        results = {
            "seqLength": seqLength,
            "maxSteps": maxSteps,
            "numEpisodes": numEpisodes,
            "average_reward": averageReward,
            "reward_std_dev": stdReward,
            "average_steps": averageSteps,
            "steps_std_dev": stdSteps
        }

        print(f"Evaluation complete for model: {modelPath.split('/')[-1]} seqLength: {seqLength} maxSteps: {maxSteps} numEpisodes: {numEpisodes}")

        return results
    
if __name__ == "__main__":
    results = []
    for seqLength in [5, 6, 7]:
        modelPath = f"./out/models/seq{seqLength}"
        currentRes = evalModel.evaluate(modelPath=f"./out/models/ppo_ai2cyber_seq{seqLength}.zip", numEpisodes=10, seqLength=seqLength)
        if currentRes is not None:
            results.append(currentRes)
            # Αυτό το είχα αρχικά για να λύσει για το 7άρι πρόβλημα αν δεν φτάσουν τα maxsteps ορισμένα ως 4*seqLength.
            # maxStepsMultiplier = 4
            # while currentRes.get("steps_std_dev") == 0.0:
            #     maxStepsMultiplier = maxStepsMultiplier * 2
            #     print(f"Warning: Steps Std Dev is 0.0 for seqLength {seqLength}. Increasing maxSteps for more reliable results.")
            #     currentRes = evalModel.evaluate(modelPath=f"./out/models/ppo_ai2cyber_seq{seqLength}.zip", numEpisodes=10, seqLength=seqLength, maxSteps=seqLength*maxStepsMultiplier, verbose=True)
            #     if currentRes is not None:
            #         results.append(currentRes)
    evalModel.writeResults(resultsFile=f"./out/evaluation_results.json", results=results)