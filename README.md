# Seafloor Mapping Toy Problem

## Objective:
A robot is tasked with mapping a 10 x 10 seafloor grid. The mission's goal is to maximize the mapped area within a fixed time limit (max_steps). 

## Grid Constraints:
  - Grid Size: The grid is 10x10 (100 cells total).

## Robot Movement:
  - The robot can move up, down, left, or right, or diagonally.
  - The robot must start from the top left of the grid.

## Not Programmed:

## Objective:
A team of robots is tasked with mapping a seafloor grid. The team must deploy between 1 to 4 robots, ensuring that at least 1 mapping robot is included in the deployment. The robots must navigate a grid to map cells, avoid failure, communicate with each other, and manage resources (battery and communication) effectively. The robots can choose to recharge at designated charging stations and may need to rescue one another in case of failure. The mission's goal is to maximize the mapped area and minimize penalties while ensuring that all assigned tasks are completed, all within a fixed time limit. 

## Scoring:
1. Mapped Cells:
  - Each successfully mapped cell is worth +1 point.
  - Any unmapped cells at the end of the mission incur a -1 point penalty per cell.

Robot Types and Roles:
At least 1 mapping robot is required. All other robot types are optional, and the team can choose which types to deploy, if any.

1. Mapping Robot:
   - Mapping Speed: Each mapping robot can map 5 cells per battery charge.
   - Battery Life: A robot's battery lasts for 5 steps before needing to recharge (each step corresponds to 1 movement on the grid).
   - Recharging: A robot must return to a charging station to recharge. Recharging takes 10 steps (a robot will be inactive during this time).

2. Standby Robot:
   - Ready to take over a failed robot’s task.
   - Can perform mapping or rescue operations but is typically in a waiting state until needed.
   - Battery Life and mapping speed are identical to the mapping robot.

3. Rescue Robot:
   - Takes over the task of a failed robot.
   - Can use the data already collected by the failed robot, avoiding the need to restart the mapping process.
   - Battery Life and mapping speed are identical to the mapping robot.

4. Buddy Robot:
   - Stays within communication range of other robots to ensure that data is shared between them.
   - Can assist in rescuing robots and can also collect mapping data for robots within range.
   - Battery Life and mapping speed are identical to the mapping robot.

5. Data Collection Robot:
   - Collects and stores data from other active robots. They do not map the grid but ensure that data is preserved.
   - Battery Life and mapping speed are identical to the mapping robot.

6. Mothership Robot:
   - A central command and communication hub for the mission.
   - It can monitor the status of other robots, oversee the mission, collect data, and manage battery levels.
   - It does not rescue other robots but can coordinate the mission, helping to keep robots within communication range and tracking their progress.
   - Battery Life: Lasts for 10 steps before needing to recharge.

Deployment:
1 to 4 robots can be deployed. The team must include at least 1 mapping robot. The remaining robots can be deployed in any of the optional roles (e.g., standby, rescue, buddy, data collection, mothership, etc.).

Key Dynamics and Constraints:
1. Battery:
   - Each robot has a limited battery capacity:
     - Mapping robots can move 5 steps and map 5 cells per charge.
     - The mothership robot lasts 10 steps.
   - Robots must manage their battery life effectively and may need to return to a charging station to recharge.

2. Charging Station:
   - Charging stations are placed at strategic locations on the grid (e.g., at the center of the grid, grid coordinates (5, 5)).
   - Robots can return to a charging station to recharge when their battery is low.
   - Recharging takes 10 steps (a robot will be inactive during this time).

3. Failure:
   - Robots have a 10% chance of failure at any point during the mission.
   - If a robot fails, it can be rescued by another robot.
   - If the robot is not rescued, the remaining robots must redo the failed robot’s task.

4. Rescue:
   - A robot can rescue a failed robot. If it does, it can use the data collected by the failed robot, avoiding the need to restart the mapping process.
   - The rescuer robot is awarded a +10 point bonus for saving the mission data and keeping things on track.
   - The mothership robot does not have the ability to rescue other robots.

5. Communication:
   - Robots can only communicate with each other if they are within a certain communication range (e.g., 3 cells).
   - The mothership robot can serve as a communication relay for robots that are out of range, helping to coordinate actions and share data between distant robots.

6. Failure Penalty:
   - If two robots fail simultaneously, there is a -20 point penalty due to the failure cascading into a larger operational issue.


Scoring:
1. Mapped Cells:
   - Each successfully mapped cell is worth +1 point.
   - 2. Unmapped Cells: Any unmapped cells at the end of the mission incur a -1 point penalty per cell.

2. Charging:
   - Returning to a charging station incurs a time penalty since the robot is inactive during recharging.
   - Recharging takes 10 steps during which the robot is inactive.

3. Failure Penalties:
   - If a robot fails and is not rescued, the mission incurs a -20 point penalty due to the lost time and effort.

4. Rescue Bonus:
   - A robot that rescues another robot is awarded a +10 point bonus for maintaining the mission’s progress and avoiding data loss.

5. Unmapped Cells:
   - Any unmapped cells at the end of the mission incur a -1 point penalty for each cell.

Strategic Decisions:
- How many robots to deploy and in which roles (mapping, rescue, standby, buddy, data collection, mothership).
- When to recharge each robot, balancing recharging time with the need to keep mapping.
- When to rescue a failed robot, weighing the risk of having to redo work versus saving time by using the data already collected.

** Question – how do the robots know if they can communicate? Is there some way they can determine if they’re in range from another robot?

