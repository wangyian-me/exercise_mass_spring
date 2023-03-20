# exercise_mass_spring
it's a course project for "Physics Simulation for Computer Graphics"

## Env

I use python3.8 environment with taichi 1.4.1.

## Results

I implemented explicit euler, implicit euler, symplectic euler, and explicit midpoint method.
Since implicit midpoint method is the same as implicit euler with half timesteps, I haven't implement it for this time.

To run the code, simply run ``` python mass_spring_main.py ```.

I setup some args to switch the mode, ``` python mass_spring_main.py --mode implicit``` would give the result of implicit euler. Similiarly, ```--mode [explicit/midpoint/symplectic]``` would give the results under different mode respectively.
