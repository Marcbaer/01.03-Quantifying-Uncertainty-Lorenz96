# 01.03-Quantifying-Uncertainty-Lorenz96

This experiment evaluates GP-LSTM's (Gaussian Process regression in combination with LSTM's) on their ability to forecast the predictive distribution of high dimensional chaotic systems.
The GP-LSTM models are built using the keras-gp library (https://github.com/alshedivat/keras-gp) with a matlab engine.

Please check the [README_Lorenz96 File](README_Lorenz96.docx) for detailed instructions on how to run the experiment.
The code was deployed on a computer cluster (ETH Euler) and is implemented to predict and propagate the dimensions of the dynamical system in parallel. MPI is used for message passing between nodes.

*01.03 High Dimensional Lorenz 96 System*

The GP-LSTM architecture is applied for forecasting and uncertainty quantification in the Lorenz 96 model. Lorenz 96 is a chaotic dynamical system developed by Edward N. Lorenz in [34] and describes the behavior in the mid-latitude atmosphere.
Hence, it is often used as a benchmark for weather forecasting methods. 
The differential equations which describe the dynamics of the system
are given by:

<img src="https://render.githubusercontent.com/render/math?math=\frac{\partial X_{j}}{\partial t} \ = \left(X_{j+1}-X_{j-2}\right)X_{j-1}-X_{j}"> +F

The parameter *F* defines the positive external forcing term and *J* the total number of Lorenz states, hence the number of dimensions.

<img src="./Figures/L3D_uncertainty10.jpg">

