# Timestep prediction using machine learning

This repository is designed to estimate the number of times a timestep length
should be halved in order for a given solver to attain convergence. For solvers
with adaptive timestepping functionality, this allows the user to predict the
timestep length that can be used at each step in the timestepping scheme. If
this prediction is accurate then it avoids the need for trial-and-error
approaches, whereby successively halved timestep lengths are tried until the
solver converges.
