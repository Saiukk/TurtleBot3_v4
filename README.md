![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![Unity](https://img.shields.io/badge/unity-%23000000.svg?style=for-the-badge&logo=unity&logoColor=white)
![C#](https://img.shields.io/badge/c%23-%23239120.svg?style=for-the-badge&logo=csharp&logoColor=white) 
![Ubuntu](https://img.shields.io/badge/Ubuntu-E95420?style=for-the-badge&logo=ubuntu&logoColor=white)

# TurtleBot4 Project

## Requirements & Installation
Ubuntu 20.04 version is required or VM (probably lagging)
Execute following commands in Installation folder:

```shell
./install_unityHub.sh
./install_mono.sh
./install_ros2_foxy.sh
```

## Create the model
Go in TurtleBot3_v4/MobileRoboticsDQN/DQN, in this repo you can find the executable file to training and save your model and test it after that.

### Training
In "training.py" you can start your training phase. 

```python
algo = PPO_PT(env, discrete=True)
algo.loop(10000) #Change to add or remove episodes
```
Here you can change you algo, choose between DQN, Reinforce and PPO and the number of episodes of training.

Execute the file and start "TurtleBot3UnityDQN" project in Unity

### Testing
Load your saved model in "testing.py" and execute the script. Now you can visualize how your model perform in the environment with results

## Simulation (ROS)

For simulation with ros, move "ROS2Package/TurtleBot3_Sim" in colcon_ws repository in your home (this folder was created using installation scripts)

### Usage
- Open "Turtlebot3UnityRos2" project in Unity
- Start Unity game
- Open your terminal in "colcon_ws" directory and digit following commands:

```shell
colcon build
source install/setup.bash
ros2 run turtlebot3_Sim turtlebot3_Sim 
```

## Real
Create connection with your turtlebot with ssh and repeat same commands mentioned in usage but change "ROS_DOMAIN_ID" from 0 to 30 in bashrc
