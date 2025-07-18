# Lindblad Modelling of a SQUID
Final Year Undergraduate Project

This repository contains code and notes related to the quantum modelling of a SQUID (Superconducting Quantum Interference Device) using the Lindblad master equation formalism. The project explores both steady-state and time-dependent dynamics under dissipative evolution.

## Overview
The goal of this project is to simulate the quantum behavior of a SQUID in open systems, accounting for decoherence and dissipation using Lindblad operators. It covers:

1. Steady-State Analysis:
Solving the Lindblad master equation to find the steady-state density matrix and observables.

2. Time-Dependent Dynamics:
Simulating the transient evolution of the system under various initial conditions and external driving.

## Key Features
1. Lindblad master equation solver using numerical integration
2. Support for custom Hamiltonians and dissipation channels
3. Visualization tools for density matrix evolution, expectation values, and coherence

## Requirements
The project is written in Python. You will need the following libraries:

1. numpy
2. scipy
3. matplotlibq

## File Structure
steady_state.py – Code for finding steady-state solutions

time_dynamics.py – Time evolution simulations

hamiltonian.py – Hamiltonian and Lindblad operator definitions

plots/ – Visualizations and figures used in the project

notes/ – Supplementary notes and derivations (if included)

## Usage
Modify parameters in the scripts to explore different regimes (e.g., coupling strength, detuning, temperature). Each script is documented for clarity and reproducibility.

## Acknowledgments
This work was completed as part of the final year undergraduate physics project. Thanks to Dr. Stephen Duffus for guidance and support.

## License
Open for academic and non-commercial use.

Bachelors Thesis - https://drive.google.com/file/d/1w4NweHPtCMsxwigGp6w92vHLy93bRqcn/view?usp=sharing
