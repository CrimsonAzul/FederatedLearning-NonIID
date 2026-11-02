## Results

Mean ± std over seeds of the test accuracy averaged over each run's last five evaluations. Centralized and local-only rows count epochs instead of rounds.

| Run | Data split | Clients | Final test accuracy (%) | Rounds to 60% | Seeds |
|---|---|---|---|---|---|
| Centralized | pooled (all data) | 1 | 86.5 ± 0.3 | 2 epochs | 3 |
| Local only | IID (α = 100) | 10 | 68.1 ± 0.1 | 15 epochs | 3 |
| Local only | Dirichlet α = 1 | 10 | 55.7 ± 2.1 | not reached | 3 |
| Local only | Dirichlet α = 0.1 | 10 | 25.5 ± 0.9 | not reached | 3 |
| FedAvg | IID (α = 100) | 10 | 87.6 ± 0.0 | 4 | 3 |
| FedAvg | Dirichlet α = 1 | 10 | 85.9 ± 0.3 | 7 | 3 |
| FedAvg | Dirichlet α = 0.1 | 10 | 77.0 ± 1.1 | 21 | 3 |

### Headline numbers

- Centralized upper bound: 86.5%
- FedAvg, 10 clients: 87.6% at α = 100 → 77.0% at α = 0.1 (-10.5 points)
- Local-only lower bound at α = 0.1: 25.5%
