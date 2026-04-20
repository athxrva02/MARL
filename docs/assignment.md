# Research Question: Credit Assignment

Question: When extraction fails in the multi-agent pipeline, how should credit (or blame) be
assigned to individual agents to enable effective learning?
Approaches to evaluate:

- End-to-end: Single reward signal for entire pipeline (simple but noisy)
- Cascading confidence: Each agent outputs confidence; credit = product of
confidences 
- Counterfactual: What would have happened with different agent outputs?
Success metric: Learning speed and final performance compared to end-to-end baseline


# Research Methodology
Literature Review
Survey existing work on:
- Credit assignment in multi-agent reinforcement learning
- Multi-armed bandits and Thompson sampling in production systems
- Contextual bandits for personalization and recommendation



# Experimentation
- Build a simulation framework to test learning approaches without expensive LLM calls


# Notes
- The reinforcement learning part should allow for learning for the entire multiagent system as a whole including the Document Analysis, Log Parser, Supervisor and Code Agent (and its subagents).
- The errors could include the ones from the agents itself (FUTURE WORK, DO NOT DO NOW)
- The RL system should help the system as a whole globally
- We can generate some synthetic data to help with training
- We need to answer: What is the best way to inject information into the RL framework we will make?
- Compare baselines, our implementation and a proper statistical study
- To learn from previous mistakes and find optimal policies, we should store the errors and correct decisions in an appropriate database (Take SQLite for now).