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


