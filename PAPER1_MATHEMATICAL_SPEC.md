# Paper 1 Mathematical Specification

## Learning to Regulate Learning in Open-Ended Gridworlds

This document is the mathematical specification for Paper 1. It defines the environment, the inner learning process, the outer meta-controller, the transfer-aware candidate evaluator, the objective, the baselines, the metrics, and the reproducibility protocol.

The specification is intentionally aligned with the current implementation. Symbols are introduced before they are used, and implementation-specific choices are stated explicitly so that the method can be audited and reproduced.

## 1. Research question

The agent does not receive a fixed training algorithm. It receives an online learning process whose regulation variables may be changed while the agent is learning.

The central question is:

Can a meta-controller choose bounded edits to the inner learner so that the learner discovers more of a changing or partially held-out gridworld, while avoiding edits that look good on one probe rollout but fail under transfer?

The proposed answer is a two-level system:

1. An inner agent learns to act and explore.
2. An outer controller proposes edits to the inner learning process, evaluates them through isolated counterfactual probes, and commits only a safe improvement.

The claim is about learning-process control, not about a new shortest-path planner.

## 2. Task family

Let a task be a finite gridworld MDP

M = (S, A, P, r, gamma, rho_0).

The state space S consists of the agent position and the task observation. The action set is

A = {up, down, left, right, interact}.

For a task with height H and width W, the set of free cells is F subset of {0,...,H-1} x {0,...,W-1}. The coverage of a trajectory is the fraction of free cells visited at least once:

C_t = |V_t| / |F|,

where V_t is the set of visited free cells up to environment step t.

The inner horizon is K environment steps per outer update. An experiment contains E episodes. The task seed determines the layout and the stochastic action sequence.

The implementation currently uses an 8-dimensional observation without local wall features and a 12-dimensional observation with local wall features. The transfer-aware configuration uses the 12-dimensional representation by default.

## 3. Inner learning process

At inner step t, the agent receives observation o_t, chooses action a_t according to a policy pi_theta, receives reward r_t, and observes o_(t+1).

The agent maintains:

- theta: online Q-network parameters;
- theta_minus: target Q-network parameters;
- D: uniform replay memory;
- P: prioritized replay memory used by the meta-control runner;
- W: working memory;
- R: random-network-distillation module;
- epsilon_base(t): base exploration schedule;
- alpha_t: optimizer learning rate;
- beta_t: curiosity or intrinsic-reward multiplier;
- p_t: prioritized-replay exponent;
- m_t: uniform/prioritized replay mixing ratio;
- epsilon_eff(t): effective exploration rate.

The effective exploration rate is

epsilon_eff(t) = m_t epsilon_base(t).

The base schedule is the implementation's exponentially decaying epsilon schedule, while m_t is the live scalar edited by the meta-controller. The multiplier is bounded indirectly by the edit range and is initialized to 1.

## 4. Observation model

The base observation contains:

1. normalized agent row;
2. normalized agent column;
3. normalized row displacement to the nearest non-empty object;
4. normalized column displacement to the nearest non-empty object;
5. a three-way one-hot object type indicator;
6. normalized current step.

The two coordinates and two displacements contribute four scalars, the object
type contributes three scalars, and the step contributes one scalar, giving
eight dimensions. Coverage, novelty, working-memory loss, RND reward, and TD
error are not part of the environment observation; they enter the separate
outer metric state below.

With transfer features enabled, four additional binary features are appended:

9. wall or out-of-bounds indicator for up;
10. wall or out-of-bounds indicator for down;
11. wall or out-of-bounds indicator for left;
12. wall or out-of-bounds indicator for right.

Thus

o_t in R^8

for the original representation, and

o_t in R^12

for the transfer-aware representation.

The four local-wall features expose a compact task-local affordance signal. They do not reveal the complete layout, visited map, or future trajectory.

## 5. Inner objective

The inner agent optimizes a bootstrapped value-learning objective. For a sampled replay transition (o_t, a_t, r_t, o_(t+1), d_t), define

y_t = r_t + gamma_eff(t) (1 - d_t) max_a Q_(theta_minus)(o_(t+1), a),

where d_t is one at a terminal transition and zero otherwise.

The temporal-difference error is

delta_t = y_t - Q_theta(o_t, a_t).

The prioritized replay loss is

L_TD(theta) = E_(i ~ p(i)) [ w_i delta_i^2 ],

where p(i) is the replay sampling distribution and w_i is the importance correction weight.

The implementation also uses working-memory prediction loss and random-network-distillation signals. These signals affect the learner and the controller's metric state; they are not claimed to be a separately optimized supervised task.

## 6. Discovery and curiosity signals

Let V_t be the set of cells visited by time t. Define the one-step coverage gain

Delta C_t = C_(t+1) - C_t.

Let n_t denote a novelty signal derived from recent observations and memory. Let q_t^RND denote the random-network-distillation prediction error or normalized intrinsic reward.

The implementation reports a composite inner reward for candidate evaluation using coverage gain, movement, and working-memory loss:

r_inner,t = Delta C_t + 0.001 I[a_t changes position] - 0.1 ell_WM,t.

This reward is used to compare learning-process edits during short counterfactual probes. It is not a replacement for the environment's task reward.

## 7. Regulation edit space

The outer controller edits the following five coordinates:

u = (u_1, u_2, u_3, u_4, u_5).

Their meanings are:

u_1: curiosity or intrinsic-reward multiplier;
u_2: optimizer learning-rate multiplier;
u_3: prioritized-replay exponent;
u_4: effective exploration multiplier;
u_5: uniform/prioritized replay mixing ratio.

The legal box is

U = [0.1, 2.0] x [0.5, 2.0] x [0, 1] x [0.01, 1] x [0, 1].

An edit is always clipped coordinatewise to U:

Pi_U(x)_j = min(max(x_j, l_j), h_j).

The live edit mapping is:

beta_t <- u_1,
alpha_t <- alpha_base u_2,
p_t <- u_3,
epsilon_eff(t) <- u_4 epsilon_base(t),
m_t <- u_5.

The discount factor gamma in the inner Bellman target is not one of the five
live regulation coordinates in this runner. Working-memory loss is measured
and reported by the probe objective, but its weight is not a live edit in the
current Paper 1 implementation.

The learning-rate multiplier is applied to a fixed base learning rate at each update. This prevents repeated live edits from compounding the learning rate unintentionally.

## 8. Meta-controller input

The outer controller receives a metric state z_t. The current implementation includes:

z_t = [
C_t,
epsilon_eff,t or epsilon_cortex,t,
eta_t,
mean_50(delta_t),
mean_50(ell_WM,t),
mean_50(q_t^RND),
trend_50(C_t),
alpha_t
].

The trailing-window operator mean_50 is the mean over the most recent 50 recorded values, with shorter history used during warm-up. The coverage trend is the recent slope or difference statistic maintained by the executive layer.

The metric state is a compact summary, not a complete Markov state. This is an explicit limitation: the outer problem is generally partially observed.

## 9. Candidate proposal policy

The learned outer policy is a small multilayer perceptron

p_phi(u | z_t).

The implementation proposes bounded edits by combining a policy prediction with bounded noise:

u^(k) = Pi_U(mu_phi(z_t) + sigma xi^(k)),

where xi^(k) is a zero-mean noise vector and sigma is the proposal scale. The current proposal scale is 0.12.

The current edit is retained as an incumbent candidate. This provides an identity action and makes the selection step conservative.

The candidate set at outer iteration t is

B_t = {u_inc,t} union {u_t^(1), ..., u_t^(K)}.

The learned policy is updated from elite candidates. If E_t is the set of the top fraction of candidates, the update minimizes

L_meta(phi) = (1 / |E_t|) sum_(u,y) in E_t (f_phi(z_t) - u)^2,

where f_phi is the controller's deterministic edit prediction and y is the selected candidate target. This is a simple elite-regression update, not a claim of globally optimal policy-gradient control.

## 10. Isolated counterfactual evaluation

A candidate must be evaluated without contaminating the live learning process.

Let x_t denote the complete live snapshot:

x_t = (environment, agent, target network, replay memory, RND state, working memory, executive state).

For candidate u, the evaluator creates a deep copy

x_t^(u) = DeepCopy(x_t),

applies u to that copy, and runs K inner steps. The live state x_t is unchanged during evaluation.

The per-task counterfactual reward is

J_i(u) =
C_final,i(u) - C_start,i
+ 0.001 M_i(u)
- 0.1 mean_t ell_WM,t^(i)(u),

where M_i(u) is the number of movement transitions during probe task i.

The current task uses the live environment and current observation in the first probe. Additional probes instantiate fresh layouts using independent seeds. This gives a candidate a task set

T_t = {T_1, ..., T_N},

where N is eval_probe_tasks.

The evaluator returns the vector

J(u) = [J_1(u), ..., J_N(u)].

The candidate information log stores task-level rewards and summary statistics. This is required for diagnosing transfer failures.

## 11. Risk-sensitive transfer objective

A candidate that performs well on average but fails badly on one probe should not automatically replace the incumbent. The robust score is

R(u) = mean_i J_i(u)
       - rho std_i(J_i(u))
       + kappa min_i J_i(u),

where:

- rho >= 0 is the variance-risk penalty;
- kappa >= 0 weights worst-case performance;
- std_i is the sample standard deviation when there is more than one probe;
- min_i is the worst observed probe reward.

The current transfer-aware defaults are

rho = 0.5,
kappa = 0.5.

For one probe, the standard-deviation term is zero. Therefore, increasing eval_probe_tasks is necessary to identify cross-layout variance.

The outer selection rule is

u*_t = argmax_(u in B_t) R(u).

The corresponding outer reward is

r_outer,t = R(u*_t).

## 12. Incumbent protection and trust region

The incumbent is the edit currently applied to the live system. Let u_inc be the incumbent and let delta be the required minimum improvement.

The controller commits the proposed winner only if

R(u*_t) >= R(u_inc) + delta.

Otherwise it commits the incumbent:

u_commit,t =
  u*_t, if R(u*_t) >= R(u_inc) + delta,
  u_inc, otherwise.

The current runner uses delta = 0.005 for learned control. This is a score-space trust region: the candidate must demonstrate a meaningful robust-score improvement before changing the live learning dynamics.

The incumbent is also included in the candidate set so that the controller can preserve a known-good edit when all noisy proposals are worse.

## 13. Fixed and random baselines

The fixed baseline applies the same physical edit at every outer step:

u_fixed = (1.0, 1.0, 0.6, 1.0, 0.5).

The random baseline samples a legal edit without using the learned policy. It is useful for measuring whether learned selection is better than arbitrary intervention.

The learned controller and baselines share:

- the same environment family;
- the same episode count;
- the same seed list;
- the same inner agent architecture;
- the same observation representation;
- the same evaluation protocol;
- the same held-out layout protocol.

Only the outer regulation strategy should differ.

## 14. Evaluation metrics

For seed s and episode e, let C_(s,e) be final training coverage.

Final coverage is

FinalCoverage_s = mean over the final evaluation window of C_(s,e).

Coverage area under the learning curve is

AUC_s = (1 / E) sum_(e=1)^E C_(s,e).

For held-out tasks h, report both:

HeldoutGreedy_s = mean_h C_h^(greedy),
HeldoutExploratory_s = mean_h C_h^(exploratory).

The transfer gap can be reported in absolute form:

Gap_abs = FinalCoverage_train - HeldoutGreedy,

or as a normalized relative gap:

Gap_rel = 1 - HeldoutGreedy / max(FinalCoverage_train, epsilon).

A lower gap is better only when training performance remains comparable. Therefore, transfer should be reported jointly with training coverage, AUC, and exploratory held-out coverage.

For n seeds, report the mean

mean(x) = (1/n) sum_s x_s,

the sample standard deviation

sd(x) = sqrt((1/(n-1)) sum_s (x_s - mean(x))^2),

and preferably a confidence interval or bootstrap interval once the final seed count is selected.

## 15. Recommended statistical protocol

The minimum development check uses three seeds. A paper result should use at least ten independent seeds for the main comparison and should keep the seed list fixed across controllers.

For each controller:

1. Train for the same number of episodes.
2. Record the complete learning curve.
3. Evaluate the same held-out layout seeds.
4. Report mean, standard deviation, and per-seed values.
5. Use paired seed-wise comparisons where possible.
6. Report the number of probe tasks and all risk coefficients.
7. Report both greedy and exploratory transfer.

The main comparison should include learned, fixed, and random controllers. An ablation table should additionally include:

- no incumbent protection;
- mean-only candidate scoring;
- one probe task;
- no local-wall features;
- no learned outer updates;
- no transfer-aware objective.

## 16. Current development evidence

The current three-seed transfer-aware development comparison used local-wall features, three probe tasks, rho = 0.5, kappa = 0.5, and 100 episodes.

Learned controller:

- final coverage mean: 0.26998;
- final coverage standard deviation: 0.00771;
- held-out greedy coverage mean: 0.03652;
- held-out exploratory coverage mean: 0.05098;
- mean AUC across the three reported seeds: approximately 0.26032.

Matched fixed controller:

- final coverage mean: 0.23860;
- final coverage standard deviation: 0.00698;
- held-out greedy coverage mean: 0.02868;
- held-out exploratory coverage mean: 0.05000;
- mean AUC across the three reported seeds: approximately 0.23172.

These numbers are development evidence, not final paper claims. They are too small a sample for a definitive statistical conclusion. The final paper should rerun the comparison with the pre-registered ten-seed protocol and longer training.

## 17. Complete project build

This specification is the source of truth for the Paper 1 build. The project
is a normal installable Python package defined by `pyproject.toml`. A clean
build consists of dependency installation, package installation, syntax
verification, invariant verification, and the matched controller experiment.

From the repository root:

~~~text
python -m pip install -e .
python -m py_compile gridworld_track/gridworld.py gridworld_track/train_meta_control.py verify/verify_meta_controller.py
python -m verify.verify_meta_controller
~~~

To create a distributable wheel:

~~~text
python -m pip wheel --no-deps . --wheel-dir dist
~~~

The implementation-to-specification build has these required entry points:

- `gridworld_track.train_meta_control`: learned, fixed, and random controllers;
- `verify.verify_meta_controller`: proposal bounds, isolation, risk-score, and incumbent checks;
- `PAPER1_MATHEMATICAL_SPEC.md`: equations, defaults, metrics, and paper protocol.

The package build must succeed before paper experiments are considered
reproducible. Generated `build/`, `dist/`, and `*.egg-info/` directories are
artifacts and should not be treated as source files.

## 18. Reproducibility commands

The invariant and snapshot-isolation checks are:

~~~text
python -m verify.verify_meta_controller
~~~

A transfer-aware learned-controller sweep is:

~~~text
python -m gridworld_track.train_meta_control --episodes 300 --controller learned --seeds 0,1,2,3,4,5,6,7,8,9 --eval-probe-tasks 3 --risk-penalty 0.5 --worst-case-weight 0.5 --transfer-features
~~~

The matched fixed baseline is:

~~~text
python -m gridworld_track.train_meta_control --episodes 300 --controller fixed --seeds 0,1,2,3,4,5,6,7,8,9 --eval-probe-tasks 3 --risk-penalty 0.5 --worst-case-weight 0.5 --transfer-features
~~~

The original observation condition can be run with:

~~~text
python -m gridworld_track.train_meta_control --episodes 300 --controller learned --seeds 0,1,2,3,4,5,6,7,8,9 --no-transfer-features
~~~

The exact command-line defaults should be recorded with every result artifact.

## 19. Implementation correspondence

| Mathematical object | Implementation location |
| --- | --- |
| Gridworld MDP and coverage | gridworld_track/gridworld.py |
| Inner agent and effective exploration | core/agent.py |
| Working memory and replay-related state | core/memory.py and core/seal/regulation.py |
| Regulation edit bounds | core/seal/regulation.py |
| Learned candidate policy | core/seal/meta_controller.py |
| Snapshot and live probe evaluator | gridworld_track/train_meta_control.py |
| Risk-sensitive score | gridworld_track/train_meta_control.py |
| Sweep aggregation | gridworld_track/train_meta_control.py |
| Invariant verification | verify/verify_meta_controller.py |

## 20. Claims the paper may make

Subject to the final ten-seed experiments, the paper may claim:

1. A bounded meta-controller can regulate several inner-learning variables online.
2. Isolated snapshot evaluation allows candidate comparison without corrupting the live learner.
3. Risk-sensitive multi-layout scoring explicitly penalizes fragile edits.
4. Incumbent protection prevents a noisy candidate from replacing a stronger current edit without a meaningful improvement.
5. The transfer-aware observation and evaluation protocol makes the train-to-held-out gap measurable.

The paper should not claim:

- universal open-ended intelligence;
- optimal regulation;
- statistically conclusive superiority from three seeds;
- transfer to arbitrary environments;
- that the controller discovers causal explanations for every edit.

## 21. Limitations and next experiments

The current study is limited by small gridworlds, short probe horizons, a compact metric state, and a modest number of candidate edits. The evaluator is computationally expensive because it performs cloned inner rollouts.

The strongest next experiments are:

1. ten or more seeds with 300 to 1000 episodes;
2. at least five held-out layouts per training seed;
3. probe-task ablations N in {1, 3, 5};
4. risk-coefficient sweeps over rho and kappa;
5. transfer to different obstacle densities and grid sizes;
6. controller-cost reporting in wall-clock time and number of probe rollouts;
7. confidence intervals and paired statistical tests;
8. an ablation of local-wall features;
9. an ablation of incumbent protection;
10. a fixed compute-budget comparison so that extra candidate evaluation is accounted for.

The mathematical contribution is therefore a testable learning-process control framework, not a promise that every component is already final.
