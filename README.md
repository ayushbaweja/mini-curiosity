# mini-curiosity


# TODOs

- New environments. Paper uses Mariobros and Vizdoom. Ideally something small that's fast to train while has lots of noises features and sparsed rewards.

- Implement the feature filter and inverse model.

- Implement A3C, ICM, ICM-Pixels (should just be ICM but without feature filtering)

- Report


# Tentative work division

### **Person 1: Environment & A3C Baseline Lead 🚀**

This person builds the foundation that everyone else will use.

* **Task 1: Environment Setup.**
    * [cite_start]Select and implement 1-2 simplified environments (e.g., ViZDoom [cite: 97] or a grid world).
    * **Crucially**: Implement the "noisy" distractor element. This is essential to show *why* ICM works. [cite_start]This could be a "TV with white noise" in a corner of the agent's view, as described in the paper[cite: 24, 105, 171].
* **Task 2: A3C Baseline Implementation.**
    * [cite_start]Implement the full Asynchronous Advantage Actor-Critic (A3C) algorithm [cite: 63] as the base policy agent.
    * [cite_start]This includes the policy and value networks, and the asynchronous training setup [cite: 118, 120-127].
    * [cite_start]Ensure this baseline agent can run and train in the noisy environment (it is expected to perform poorly or get distracted, as shown in Figure 6 [cite: 174]).

---

### **Person 2: ICM - Feature Encoder & Inverse Model 🧠**

This person implements the self-supervised feature learning part of the Intrinsic Curiosity Module (ICM).

* **Task 1: Feature Encoder ($\phi$).**
    * [cite_start]Using Person 1's A3C code, implement the feature encoding network ($\phi$) that transforms raw states ($s_t$ and $s_{t+1}$) into feature vectors[cite: 55, 81]. [cite_start]This is typically a convolutional network[cite: 129].
* **Task 2: Inverse Dynamics Model.**
    * [cite_start]Implement the inverse model, which takes the two feature vectors ($\phi(s_t)$, $\phi(s_{t+1})$) and predicts the action ($a_t$) that was taken to get from $s_t$ to $s_{t+1}$[cite: 55, 81, 130].
* **Task 3: Inverse Loss ($L_I$).**
    * [cite_start]Implement the loss function ($L_I$) for the inverse model (e.g., cross-entropy for discrete actions)[cite: 82]. [cite_start]This loss is what trains the feature encoder ($\phi$)[cite: 88].

---

### **Person 3: ICM - Forward Model & Reward Integration 🎁**

This person implements the "curiosity" reward generation and integrates it into the A3C agent.

* **Task 1: Forward Dynamics Model.**
    * [cite_start]Implement the forward model, which takes the current feature vector ($\phi(s_t)$) from Person 2's encoder and the action ($a_t$) to predict the *next* feature vector ($\hat{\phi}(s_{t+1})$)[cite: 56, 85, 131].
* **Task 2: Intrinsic Reward Generation.**
    * [cite_start]Implement the forward loss ($L_F$), which is the prediction error between the predicted next feature ($\hat{\phi}(s_{t+1})$) and the actual next feature ($\phi(s_{t+1})$)[cite: 85].
    * [cite_start]Use this prediction error as the intrinsic curiosity reward ($r_t^i$)[cite: 57, 86].
* **Task 3: Reward Integration.**
    * [cite_start]Modify Person 1's A3C agent to combine the external reward ($r_t^e$) and the new intrinsic reward ($r_t^i$) to form the total reward $r_t$ that the policy optimizes[cite: 52].

---

### **Person 4: "ICM-Pixels" Baseline Implementation 📺**

This person implements the critical baseline used to prove the value of the ICM's learned features.

* **Task 1: Pixel-Based Forward Model.**
    * [cite_start]Using Person 1's A3C base, implement a forward model that *directly predicts the next raw state (pixels)* from the current state and action[cite: 135].
    * [cite_start]This model will have a different architecture from Person 3's, likely using deconvolutional layers to reconstruct the image[cite: 143].
* **Task 2: Pixel-Based Reward.**
    * Calculate the prediction error in pixel space (e.g., mean squared error) and use this error as the intrinsic reward.
    * Integrate this reward into the A3C agent, just as Person 3 did. [cite_start]This agent is expected to get "distracted" by the noisy part of the environment, as its pixel predictions will always be wrong there[cite: 174, 175].

---

### **Person 5: Project Lead - Integration, Experimentation & Analysis 📊**

This person is responsible for the final product, testing, and analysis.

* **Task 1: Code Integration.**
    * Manage the main code repository.
    * Merge the A3C base (from P1), the full ICM (from P2 + P3), and the ICM-Pixels baseline (from P4) into a single, functional codebase.
* **Task 2: Experimentation.**
    * Run all three models (A3C, A3C+ICM, A3C+ICM-Pixels) in the noisy environment(s).
    * Methodically log all results (e.g., extrinsic rewards, episode lengths, intrinsic rewards).
* **Task 3: Analysis & Reporting.**
    * [cite_start]Plot the learning curves for all three agents, similar to Figures 5 and 6 in the paper[cite: 593].
    * [cite_start]Analyze and write up the final results, demonstrating that A3C+ICM successfully explores *despite* the noise, while A3C+ICM-Pixels gets distracted by it[cite: 174, 175].
