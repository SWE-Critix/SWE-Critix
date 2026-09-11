<h1 align="center" style="display: flex; align-items: center; justify-content: center; gap: 12px;">
  <img src="assets/images/logo.jpeg" alt="Logo" style="height: 120px;">
  <span>SWE-Critix: Post-training Generalist Verifiers for Assessing Coding Agent Trajectories</span>
</h1>

<p align="center">
  <a href="" style="text-decoration: none; font-family: Gill Sans MT; font-weight: semibold;">Jiajun Hu, </a>
  <a href="" style="text-decoration: none; font-family: Gill Sans MT; font-weight: semibold;">Chun Yong Chong</a>
</p>

<p align="center">
	<a href="https://arxiv.org/abs/xxxx.xxxxx" style="text-decoration: none;">
  		<img src="https://img.shields.io/badge/arXiv-xxxx.xxxxx-B31B1B?logo=arxiv&logoColor=white" style="height: 16px;">
  		https://arxiv.org/abs/coming_soon
	</a>
</p>

<p align="center">
	<a href="https://github.com/SWE-Critix/SWE-Critix" style="text-decoration: none;">
  		<svg height="16" width="16" viewBox="0 0 16 16" version="1.1">
  			<path fill="black" d="M8 0c4.42 0 8 3.58 8 8a8.013 8.013 0 0 1-5.45 7.59c-.4.08-.55-.17-.55-.38 0-.27.01-1.13.01-2.2 0-.75-.25-1.23-.54-1.48 1.78-.2 3.65-.88 3.65-3.95 0-.88-.31-1.59-.82-2.15.08-.2.36-1.02-.08-2.12 0 0-.67-.22-2.2.82-.64-.18-1.32-.27-2-.27-.68 0-1.36.09-2 .27-1.53-1.03-2.2-.82-2.2-.82-.44 1.1-.16 1.92-.08 2.12-.51.56-.82 1.28-.82 2.15 0 3.06 1.86 3.75 3.64 3.95-.23.2-.44.55-.51 1.07-.46.21-1.61.55-2.33-.66-.15-.24-.6-.83-1.23-.82-.67.01-.27.38.01.53.34.19.73.9.82 1.13.16.45.68 1.31 2.69.94 0 .67.01 1.3.01 1.49 0 .21-.15.45-.55.38A7.995 7.995 0 0 1 0 8c0-4.42 3.58-8 8-8Z"></path>
  		</svg>
  		https://github.com/SWE-Critix/SWE-Critix
	</a>
</p>

<p align="center">
<a href="https://huggingface.co/SWE-Critix" style="text-decoration: none;">🤗 https://huggingface.co/SWE-Critix</a>
</p>

SWE-Critix presents an LLM post-training pipeline for evaluating coding-agent trajectories (Figure 1). The trained model (verifier) takes a trajectory as input, which consists of an issue or requirement description and the agent's recorded process of implementing a solution. If the model determines that the trajectory successfully resolves the issue or fulfills the requirement, it outputs `<judgment>YES</judgment>`; otherwise, it outputs `<judgment>NO</judgment>`. Before making the binary judgment, the model also produces a chain-of-thought (CoT) rationale that justifies its decision. The rationale includes a summary of the trajectory, an analysis of the error type, and a causal analysis of the factors leading to the success or failure of the trajectory (Figure 1 - Application).

<br>

<div id="fig1" style="text-align: center;">
  <img src="./assets/images/overview.png" alt="overview" style="max-width: 80%; height: auto;">
  <p style="font-size: 0.9em; color: #666;">Figure 1: SWE-Critix Post-training Pipeline</p>
</div>

<br>

SWE-Critix adopts a three-stage post-training pipeline. First, a teacher model is prompted to analyze a pair of correct and incorrect trajectories for the same issue. Through comparative analysis, the teacher summarizes the respective procedures, key differences, error types, and factors contributing to the success or failure of each trajectory. These summaries serve as the CoT rationales for subsequent training (Figure 1 - CoT Annotation). The resulting dataset with annotated CoT rationales is then assembled into SFT training samples, which are used to fine-tune the model (Figure 1 - SFT). Finally, SWE-Critix applies reinforcement learning to the remaining trajectories without CoT annotations. The unit testing results of the trajectories are used as the ground truth for computing the reward signal (Figure 1 - RL).

We open source the model weights, datasets, and training scripts 
- [SFT checkpoint](https://huggingface.co/SWE-Critix/SWE-Critix-Qwen3-30B-A3B-SFT-Epoch3) and [RL checkpoint](https://huggingface.co/SWE-Critix/SWE-Critix-Qwen3-30B-A3B-RL-Step-3396)
- [SFT dataset](https://huggingface.co/datasets/SWE-Critix/alpaca_style_sft_dataset), [RL dataset](https://huggingface.co/datasets/SWE-Critix/rl_dataset), and [Test dataset](https://huggingface.co/datasets/SWE-Critix/test_dataset)
- [Training scripts](https://github.com/SWE-Critix/SWE-Critix)