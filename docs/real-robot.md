# Real robot {#real-robot}

There are two real-robot spines: the mock spine, and the pi3hat spine.

## Mock spine {#mock-spine}

The mock spine is useful to run an agent on the robot without firing up the actuators. It works exactly as the pi3hat spine, replacing "pi3hat" with "mock" in all instructions.

## Pi3hat spine {#pi3hat-spine}

The pi3hat spine is the one that runs on the real robot, where a [pi3hat r4.5](https://mjbots.com/products/mjbots-pi3hat-r4-5) is mounted on top of the onboard Raspberry Pi computer. To run this spine, you can either download it from GitHub, or build it locally from source.

### Download the latest release

Assuming your robot is connected to the Internet, you can get the latest pi3hat spine from GitHub using the `upkie_tool` command-line utility:

```console
$ ssh user@upkie
user@upkie:~$ upkie_tool update
```

Once the spine is installed, start it from a command-line on the robot:

```console
user@upkie:~$ pi3hat_spine
```

You can then run any agent in a separate shell on the robot, for example the wheel balancer:

```console
user@upkie:upkie$ make run_pid_balancer
```

### Build from source

To build the pi3hat spine locally from source, then upload it to Raspberry Pi, use the Makefile at the root of the repository:

```console
make build
make upload UPKIE_NAME=your_upkie
```

Next, log into the Pi and run a pi3hat spine:

```console
$ ssh user@upkie
user@upkie:~$ cd upkie
user@upkie:upkie$ make run_pi3hat_spine
```

Once the spine is running, you can run any agent in a separate shell on the robot, for example the wheel balancer:

```console
user@upkie:upkie$ make run_pid_balancer
```
