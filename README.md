# mini-curiosity


# TODOs Summary

- Find New environments. Paper uses Mariobros and Vizdoom. Ideally something small that's fast to train while has lots of noises features and sparsed rewards.

- Implement the feature filter and inverse model.

- Implement A3C, ICM, ICM-Pixels (should just be ICM but without feature filtering)

- Report


# TODOS Breakdown
### **Environment & A3C Baseline**

* **Task 1: Environment Setup.**
    * Select and implement 1-2 simplified environments.
    * **Crucially**: Implement the "noisy" distractor element. This is essential to show *why* ICM works. This could be a "TV with white noise" in a corner of the agent's view, as described in the paper.
* **Task 2: A3C Baseline Implementation.**
    * Implement the full Asynchronous Advantage Actor-Critic (A3C) algorithm as the base policy agent.
    * This includes the policy and value networks, and the asynchronous training setup.
    * Ensure this baseline agent can run and train in the noisy environment (it is expected to perform poorly or get distracted).

---

### **ICM - Feature Encoder & Inverse Model**

Implements the self-supervised feature learning part of the Intrinsic Curiosity Module (ICM).

* **Task 1: Feature Encoder ($\phi$).**
    * Using A3C code, implement the feature encoding network ($\phi$) that transforms raw states ($s_t$ and $s_{t+1}$) into feature vectors. This is typically a convolutional network.
* **Task 2: Inverse Dynamics Model.**
    * Implement the inverse model, which takes the two feature vectors ($\phi(s_t)$, $\phi(s_{t+1})$) and predicts the action ($a_t$) that was taken to get from $s_t$ to $s_{t+1}$.
* **Task 3: Inverse Loss ($L_I$).**
    * Implement the loss function ($L_I$) for the inverse model (e.g., cross-entropy for discrete actions). This loss is what trains the feature encoder ($\phi$).

---

### **ICM - Forward Model & Reward Integration**

Implements the "curiosity" reward generation and integrates it into the A3C agent.

* **Task 1: Forward Dynamics Model.**
    * Implement the forward model, which takes the current feature vector ($\phi(s_t)$) from the encoder and the action ($a_t$) to predict the *next* feature vector ($\hat{\phi}(s_{t+1})$).
* **Task 2: Intrinsic Reward Generation.**
    * Implement the forward loss ($L_F$), which is the prediction error between the predicted next feature ($\hat{\phi}(s_{t+1})$) and the actual next feature ($\phi(s_{t+1})$).
    * Use this prediction error as the intrinsic curiosity reward ($r_t^i$).
* **Task 3: Reward Integration.**
    * Modify  A3C agent to combine the external reward ($r_t^e$) and the new intrinsic reward ($r_t^i$) to form the total reward $r_t$ that the policy optimizes.

---

### **"ICM-Pixels" Baseline Implementation ** (Optional)

Implements the critical baseline used to prove the value of the ICM's learned features.

* **Task 1: Pixel-Based Forward Model.**
    * Using A3C base, implement a forward model that *directly predicts the next raw state (pixels)* from the current state and action.
    * This model will have a different architecture from ICM, likely using deconvolutional layers to reconstruct the image.
* **Task 2: Pixel-Based Reward.**
    * Calculate the prediction error in pixel space (e.g., mean squared error) and use this error as the intrinsic reward.
    * Integrate this reward into the A3C agent, just as ICM did. This agent is expected to get "distracted" by the noisy part of the environment, as its pixel predictions will always be wrong there.

---

### **Integration, Experimentation & Analysis **

Final product, testing, and analysis.

* **Task 1: Code Integration.**
    * Manage the main code repository.
    * Merge the A3C base (from P1), the full ICM (from P2 + P3), and the ICM-Pixels baseline (from P4) into a single, functional codebase.
* **Task 2: Experimentation.**
    * Run all three models (A3C, A3C+ICM, A3C+ICM-Pixels) in the noisy environment(s).
    * Methodically log all results (e.g., extrinsic rewards, episode lengths, intrinsic rewards).
* **Task 3: Analysis & Reporting.**
    * Plot the learning curves for all three agents, similar to Figures 5 and 6 in the paper.
    * Analyze and write up the final results, demonstrating that A3C+ICM successfully explores *despite* the noise, while A3C+ICM-Pixels gets distracted by it.
