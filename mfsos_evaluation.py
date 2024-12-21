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
    observation,_ = env.reset()
    action = 0.0 * env.action_space.sample()
    for step in range(1_000_000):
        pitch = observation[0]
        control = Kp * pitch
        action[0] = control
        if force != 0:
            apply_sagittal(env, force, step, duration=1.0)
        observation, reward, terminated, truncated, _ = env.step(action)

        if terminated:
            stop_time = time.time()
            return stop_time - start_time
        if time.time() > start_time + 15.:
            return 15.
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
        if time.time() > start_time + 15.:
            return 15.
        if terminated:
            stop_time = time.time()
            return stop_time - start_time
            
        if terminated or truncated:
            observation,_=env.reset()    
    return None    



if __name__ == "__main__":
    gain = np.array([10.0, 0.01, 0.2, 0.2])
    Kp =20.0
    list_forces = np.linspace(0,5.0,num=11)
    print(list_forces)
    with gym.make("UpkieGroundVelocity-v3",frequency=200.0) as env:
        env.reset()
        times = []
        
        for force in list_forces:
            
            mf_time = linear_feedback_controller(env,gain,force)
            env.reset()
            times.append(mf_time)
            
    plt.plot(list_forces[1:],times[1:])
    plt.savefig("mfsos.png", dpi=300, bbox_inches='tight')
    plt.show()
        
    
    
