# MeanFieldBrainSimulator

This repository contains models and code to simulate brain dynamics at parcel resolution. The required input is an anatomical connectivity matrix. In the data folder, a such matrix is provided, which is the mouse isocortical structural conncetivity matrix from Oh et al. (2014). There is no unique choice of mean field large scale models. You may find a review at Messe et al. (2015), which illustrates how to use such models to predict functional connectivities. In the models folder, you can currently find two models: SAR (messe et al 2015) and DMF (Deco 2014). More models will be added progressively.

General workflow: Create a model instance with some data and parameters input -> simulate model dynamics -> obtain metrics of interest from simulated data. You may compare simulated FC with some target FC to see how well the model performs, or to run some in-silico experiments.
