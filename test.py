import gymnasium as gym
import numpy as np
import upkie.envs
import time

upkie.envs.register()

           
def apply_sagittal(env,force,step):
    sag_force = np.zeros(3)
    bullet_action = {
            "external_forces":{
                "torso":{
                    "force": sag_force,
                    "local": False,
                }
            }
        }
    if step < 200:
        sag_force[0] = force
    env.unwrapped.bullet_extra(bullet_action)
    
        

def pure_linear_feedback_controller(env,Kp,force = 0):
    
    observation,_ = env.reset()
    action = 0.0 * env.action_space.sample()
    for step in range(1_000_000):
        pitch = observation[0]
        control = Kp * pitch
        action = [control,control]
        if force != 0:
            apply_sagittal(env,force,step)
        observation, reward, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            observation,_ = env.reset()

def linear_feedback_controller(env,gain,force):
    observation,_ = env.reset()
    action = 0.0 * env.action_space.sample()
    for step in range(1_000_000):
        control = np.dot(gain,observation)
        action[0] = control
        if force != 0:
            apply_sagittal(env,force,step)
        observation, reward,terminated,truncated,_ = env.step(action)
        
            
        if terminated or truncated:
            observation,_=env.reset()    
            



if __name__ == "__main__":
    
    Kp = 11.0
    gain = np.array([10.0,1.0,0.1,0])
    #Pour le pure linear MSFOS 2.0 N
    force = 2.0
    duration = 1.0
    #pure_linear_feedback_controller(Kp)
    with gym.make("UpkieGroundVelocity-v3",frequency=200.0) as env:
        env.reset()
        #pure_linear_feedback_controller(env,Kp,force)
        linear_feedback_controller(env,gain,force)
        #apply_sagittal_force(env,Kp,force,duration)
