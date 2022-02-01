# Tutorial: ASV path planning using `whelk` and `conch`

## Introduction

The purpose of this tutorial is to demonstrate the use of software repositories [whelk](https://github.com/ekrell/whelk) and [conch](https://github.com/ekrell/conch) to generate an energy efficient path for an [autonomous surface vehicle (ASV)](https://www.sciencedirect.com/topics/earth-and-planetary-sciences/surface-vehicle). 

Consider the following planning mission. You are located on a vessel in Boston Harbor and are going to deploy an ASV that will navigate to a target location. The vehicle is equipped with an [onboard controller](https://www.hindawi.com/journals/mpe/2018/7371829/) capable of maintaining a heading and desired speed to reach a target location. To effectively navigate to the target, the controller requires a plan: a sequence of waypoints that form an obstacle-free path from vehicle’s current location to the goal. The following figure shows the ASV's **start** and **goal** locations. 

![Map of Boston Harbor showing start and goal locations](figures/boston_harbor_task.png)

Clearly the vessel cannot travel in a straight line to reach the goal - Spectacle Island is in the way. 
The most obvious planning criteria is that the path should be **feasible**: the vehicle cannot navigate through obstacles. 
We also prefer the path to be, at least in some sense, **good**. That is, a path that is optimal or near-optimal for some criteria. 
A shortest-distance path is an obvious choice, but often not the best for vehicles operating in the marine environment. 
Wind, waves, and water currents can substantially impact a vessel's efficiency. 
Ideally, the vehicle would avoid going against forces that oppose it and instead take advantage of those that are headed toward the goal. 
Even if this means deviating from the shortest path, it may be much more energy efficient which is highly desirable considering the limited energy of ASVs. 

If we have a vector field of the spatio-temporal water currents over the extent of the mission, then we can plan a path that minimizes the energy expenditure of the ASV. The problem is that the data does not exist: we cannot completely predict the state of the complex marine environment. Instead, we rely on forecasts to act as our best guess as to how the currents will behave at least in the near future. For example, the [Northeast Coastal Ocean Forecast System (NECOFS)](http://fvcom.smast.umassd.edu/necofs/). In this tutorial, we will use the `whelk` repository to download NECOFS data for a specified region and time duration, then convert it to a raster (grid) format useful for path planning. 

If we have a map of Boston Harbor and the local water current forecasts, then we should be able to plan a route for energy-efficient and obstacle-free ASV navigation. In this tutorial, we will use planning software in the `conch` repository to do so. 

Now consider why the ASV is going to that goal location. In our fictitious scenario, suppose that the vehicle is going to use its onboard sensors to collect detailed data on an [eelgrass habitat at that location](http://oceans.mit.edu/news/featured-stories/mit-sea-grass-work-featured-york-times.html). Now suppose that the vehicle's energy efficient route has the vehicle navigating close to other patches of seagrass along the way. Without explicitly targeting those patches, we might get more useful data from the mission if the waypoints were slightly modified so that the ASV passes over and samples those patches. We call this **opportunistic reward-based planning.** Unlike [coverage planners](https://www.researchgate.net/publication/221071829_Towards_marine_bloom_trajectory_prediction_for_AUV_mission_planning) that seek to maximize the sampling reward within time/energy constraints, we are proposing to still perform a point-to-point planning but with slightly relaxed efficiency constraints to take advantage of nearby sampling opportunities. In this tutorial, we will demonstrate using a raster grid of reward values to achieve opportunistic reward-based planning. For simplicity, we will rely on a synthetic reward grid. But it is based on real-world applications. Consider using satellite imagery to get coarse visual dataon habitat locations, then following up with the ASV for detailed water measurements and underwater imagery. 

This tutorial will explain:

1. How to aquire the input data to represent the planning environment (map of obstacles, water current forecasts, & reward)
2. How to use **Visibility Graphs** and **Particle Swarm Optimization** to generate a solution path

## Step 1: Acquire data 

Have raster 

Use whelk to get currents that match the raster

## Step 2: Generate path






## Conclusion

We encourage you to try new things & make pull requests!

### System limitations: 

- Global... not reactive (but could be a first step)
- Assumes vehicle is a point mass, energy usage is the work to maintain heading, speed… 
- Must maintain a constant speed
- Assumes accurate forecasts (what about a [game theoretic planner](https://www.researchgate.net/publication/340065861_Game_Theoretic_Potential_Field_for_Autonomous_Water_Surface_Vehicle_Navigation_Using_Weather_Forecasts)?)
- Metaheuristics always messy... trial & error... etc
