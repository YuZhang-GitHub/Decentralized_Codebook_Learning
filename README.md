# Decentralized Interference-Aware Codebook Learning in Millimeter Wave MIMO Systems
This is the simulation code related to the following article: Y. Zhang and A. Alkhateeb, "[Decentralized Interference-Aware Codebook Learning in Millimeter Wave MIMO Systems](https://arxiv.org/abs/2401.07479)".

# Abstract of the Article
Beam codebooks are integral components of the future millimeter wave (mmWave) multiple input multiple output (MIMO) system to relax the reliance on the instantaneous channel state information (CSI). The design of these codebooks, therefore, becomes one of the fundamental problems for these systems, and the well-designed codebooks play key roles in enabling efficient and reliable communications. Prior work has primarily focused on the codebook learning problem within a single cell/network and under stationary interference. In this work, we generalize the interference-aware codebook learning problem to networks with multiple cells/basestations. One of the key differences compared to the single-cell codebook learning problem is that the underlying environment becomes non-stationary, as the behavior of one base station will influence the learning of the others. Moreover, to encompass some of the challenging scenarios, information exchange between the different learning nodes is not allowed, which leads to a fully decentralized system with significantly increased learning difficulties. To tackle the non-stationarity, the averaging of the measurements is used to estimate the interference nulling performance of a particular beam, based on which a decision rule is provided. Furthermore, we theoretically justify the adoption of such estimator and prove that it is a sufficient statistic for the underlying quantity of interest in an asymptotic sense. Finally, a novel reward function based on averaging is proposed to fully decouple the learning of the multiple agents running at different nodes. Simulation results show that the developed solution is capable of learning well-shaped codebook patterns for different networks that significantly suppress the interference without information exchange, highlighting a promising practical codebook learning solution for dense and fast deployment in future mmWave networks.

# How to reproduce the simulation results?
1. Download all the files of this repository.
2. Run `commander.py`.
3. After step 2 finishes, run `read_beams.py` to collect the learning results (saved as `Learned_codebook.mat`).
4. Run `show_codebook.m`, which will generate the learned interference-aware codebook beam patterns as attached below.

<p float="left">
  <img src="./figures/beam_1.png" alt="Beam 1" width="200"/>
  <img src="./figures/beam_2.png" alt="Beam 2" width="200"/>
  <img src="./figures/beam_3.png" alt="Beam 3" width="200"/>
  <img src="./figures/beam_4.png" alt="Beam 4" width="200"/>
</p>
<p float="left">
  <img src="./figures/beam_5.png" alt="Beam 5" width="200"/>
  <img src="./figures/beam_6.png" alt="Beam 6" width="200"/>
  <img src="./figures/beam_7.png" alt="Beam 7" width="200"/>
  <img src="./figures/beam_8.png" alt="Beam 8" width="200"/>
</p>
<p float="left">
  <img src="./figures/beam_9.png" alt="Beam 9" width="200"/>
  <img src="./figures/beam_10.png" alt="Beam 10" width="200"/>
  <img src="./figures/beam_11.png" alt="Beam 11" width="200"/>
  <img src="./figures/beam_12.png" alt="Beam 12" width="200"/>
</p>
<p float="left">
  <img src="./figures/beam_13.png" alt="Beam 13" width="200"/>
  <img src="./figures/beam_14.png" alt="Beam 14" width="200"/>
  <img src="./figures/beam_15.png" alt="Beam 15" width="200"/>
  <img src="./figures/beam_16.png" alt="Beam 16" width="200"/>
</p>

If you have any problems with generating the figure, please contact [Yu Zhang](https://www.linkedin.com/in/yu-zhang-391275181/).

# License and Referencing
This code package is licensed under a [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](https://creativecommons.org/licenses/by-nc-sa/4.0/). If you in any way use this code for research that results in publications, please cite our original article:
> Y. Zhang and A. Alkhateeb, "[Decentralized Interference-Aware Codebook Learning in Millimeter Wave MIMO Systems](https://arxiv.org/abs/2401.07479)".
