## Taxi V3
Inside this grid environment, the objective is for the taxi to pick up the passenger from a location and drop him to the desired destination. There are 500 possible states in this 5x5 grid containing information of location of taxi and location of passenger and 6 actions for the taxi (up, down, left, right, pickup and dropoff).

<img src="/img/taxi.png" width="250" >

A Q-table learning from the bellmen equation was implemented to find the optimal action in this taxi environment. The output is a learned Q-table for the optimal action to take for every possible state that the taxi and passenger can be in. 