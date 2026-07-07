# Optimised transmission loss calculation

We are interested in the problem of placing a set of sensors to minimise the probability that an adversary can pass through an area of interest without being detected. One key component of our calculation is estimating how much sound attenuates between the source (the adversary) and the receiver (our sensor). This quantity is called the transmission loss, and is usually measured in decibels.
The transmission loss can be computed by solving the Helmholtz equation. Since the speed of sound depends on depth (the relationship is known as the sound speed profile, and depends on temperature and salinity), this equation cannot be solved analytically and we must use numerical solvers.
Perhaps the best known of these solvers is called Bellhop, it can compute the transmission loss for a given source-receiver pair in ~1 second. This is far too slow for our problem, where we need to compute the transmission loss potentially hundreds of billions of times.
Our approximate solution to this problem was to use a simple analytic model (taken from the literature) to approximate the transmission loss, which is orders of magnitude faster than using Bellhop. The downside of this approach is that it is not necessarily very accurate. We’d like to improve the accuracy of the simplified model, whilst maintaining its speed.

Our current best effort looks like this:
- Generate a set of real Bellhop calculations to serve as a ground truth dataset. Each datapoint consists of the input values (source location, receiver location, and time of year) and the output value (the transmission loss).
  - These inputs are used to calculate or look up relevant quantities, such as the distance between these points, the depth between the points, the temperature and salinity profiles needed to compute the sound speed field etc.
- We can easily use our analytic model to compute a prediction for each datapoint and compare it to the Bellhop ground truth. The difference between the analytic prediction and the Bellhop value is the residual.
- We then fit a model (e.g. XGBoost) to the residuals, using all available input data to create a correction to our analytic model.
  
This project looks to improve the performance of this correction model.
