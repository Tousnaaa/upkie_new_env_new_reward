import gymnasium as gym
import numpy as np
import upkie.envs
import time
import matplotlib.pyplot as plt

upkie.envs.register()

           
def apply_sagittal(env, force, step, duration):
    sag_force = np.zeros(3)
    bullet_action = {
            "external_forces":{
                "torso":{
                    "force": sag_force,
                    "local": False,
                }
            }
        }
    if step < int(200 * duration):
        sag_force[0] = force
    env.unwrapped.bullet_extra(bullet_action)
    
        

def pure_linear_feedback_controller(env,Kp,force = 0):
    start_time = time.time()
    observation,_ = env.reset()
    action = 0.0 * env.action_space.sample()
    for step in range(1_000_000):
        pitch = observation[0]
        control = Kp * pitch
        action = [control,control]
        if force != 0:
            apply_sagittal(env, force, step, duration=1.0)
        observation, reward, terminated, truncated, _ = env.step(action)

        if terminated:
            stop_time = time.time()
            return stop_time - start_time

        if terminated or truncated:
            observation,_ = env.reset()
    return None

def linear_feedback_controller(env,gain,force):
    start_time = time.time()
    observation,_ = env.reset()
    action = 0.0 * env.action_space.sample()
    for step in range(1_000_000):
        control = np.dot(gain,observation)
        action[0] = control
        if force != 0:
            apply_sagittal(env, force, step, duration=1.0)
        observation, reward,terminated,truncated,_ = env.step(action)
        
        if terminated:
            stop_time = time.time()
            return stop_time - start_time
            
        if terminated or truncated:
            observation,_=env.reset()    
    return None    



if __name__ == "__main__":
    list_k = np.arange(1.0, 21.0)
    print(list_k)
    list_means = []
    list_std = []
    for k in list_k:
        list_times = []
        Kp = k
        gain = np.array([10.0, 1.0, 0.0, 0.1])
        #force = 10.0
        duration = 1.0
        for i in range(10):
            #pure_linear_feedback_controller(Kp)
            with gym.make("UpkieGroundVelocity-v3",frequency=200.0) as env:
                life_time = pure_linear_feedback_controller(env, Kp, 0.0)
                list_times.append(life_time)
                env.reset()
                #linear_feedback_controller(env,gain,force)
                #apply_sagittal_force(env,Kp,force,duration)
            print("Done")
            time.sleep(1)
        #print(list_times)
        list_means.append(np.mean(list_times))
        list_std.append(np.std(list_times))
    
    plt.errorbar(list_k, list_means, yerr=list_std, fmt='-o', ecolor='red', capsize=5, label="Mean ± Std Dev")
    plt.xlabel("Kp")
    plt.ylabel("Lifetime")
    plt.title("Lifetime vs Kp with Standard Deviation")
    plt.legend()
    plt.grid(True)
    plt.savefig("lifetime_vs_kp.png", dpi=300, bbox_inches='tight')
    plt.show()
