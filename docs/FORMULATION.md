# Model formulation

Notation for the scheduling models: employees $e$, roles $r$, days $d \in \{0,\dots,6\}$, shift templates
$k$ (start hour $s_k$, length $\ell_k$, pay premium $\pi_k$), week-hours $t \in \{0,\dots,167\}$.
$C(d,k)$ is the set of week-hours covered when template $k$ starts on day $d$ (wrapping at the end of the week).
$q_{rt}$ is the number of role-$r$ staff required on duty in hour $t$.

## 1. Employee-level roster (mixed-integer program) — `roster.py`

Decision variables

| Variable | Meaning |
|---|---|
| $y_{edkr} \in \{0,1\}$ | employee $e$ works template $k$ starting day $d$, in role $r$ |
| $u_{rt} \ge 0$ | uncovered staff-hours of role $r$ in hour $t$ (filled by call-in staff) |
| $o_e \ge 0$ | overtime hours of employee $e$ |

$y_{edkr}$ exists only if $e$ is qualified for $r$ (cross-training), available on day $d$ for template $k$
(day-off requests, no-night restrictions, contract shift lengths), and $k$ covers some hour with demand for $r$.

**Objective** (weekly cost)

$$
\min \sum_{e,d,k,r} w_e\,\ell_k\,\pi_k\,y_{edkr}
\;+\; \sum_e w_e(\rho-1)\,o_e
\;+\; \sum_{r,t} m\,w_r\,u_{rt}
$$

with wage $w_e$, overtime premium $\rho$ (1.5) and call-in multiplier $m$ (3). Pricing uncovered hours at the
cost of actually filling them keeps the model feasible and makes the trade-off explicit: an 8-hour shift is
only worth adding if it covers enough otherwise-uncovered hours.

**Constraints**

| Constraint | Form | Type |
|---|---|---|
| Coverage | $\sum_{(e,d,k):\,t\in C(d,k)} y_{edkr} + u_{rt} \ge q_{rt}$ for all $r,t$ with $q_{rt}>0$ | inequality |
| Floor capacity | $\sum_{(e,d,k):\,t\in C(d,k)} y_{edkr} \le \bar{q}_r$ | inequality |
| One shift per day | $\sum_{k,r} y_{edkr} \le 1$ | inequality |
| Contract hours | $H_e = \sum_{d,k,r} \ell_k\,y_{edkr}$, with $H^{\min}_e \le H_e \le H^{\max}_e$; **$H_e = 40$ when guaranteed** | equality / inequality |
| Working days | $\sum_{d,k,r} y_{edkr} \le D^{\max}_e$ | inequality |
| Minimum rest | $\sum_r y_{edkr} + \sum_r y_{e,d+1,k',r} \le 1$ whenever $24 + s_{k'} - (s_k+\ell_k) < 11$ | inequality |
| Weekend limit | $\sum_{d\in\{\text{Sat},\text{Sun}\}} \sum_{k,r} y_{edkr} \le W^{\max}$ | inequality |
| Overtime | $o_e \ge H_e - 40,\; o_e \le O^{\max}$ | inequality |

The casino scenario guarantees full-timers exactly 40 hours ($H^{\min}=H^{\max}=40$), which turns the hours
constraint into an equality.

Solved with CBC (bundled with PuLP) to a 0.5% relative gap; every bundled scenario solves to proven optimality
in under a second (the casino model has 1,737 binary and 2,291 total variables).

## 2. Aggregate covering model (LP) — `staffing_lp.py`

The classic labor-scheduling LP drops employee identities:

$$
\min \sum_{r,d,k} w_r\,\ell_k\,\pi_k\,x_{rdk} + \sum_{r,t} m\,w_r\,u_{rt}
\quad\text{s.t.}\quad \sum_{(d,k):\,t\in C(d,k)} x_{rdk} + u_{rt} \ge q_{rt},\quad x,u \ge 0 .
$$

Every feasible roster maps to a feasible $x$ with the same cost, so the LP optimum is a **lower bound** on the
roster cost; the gap between them is the price of individual rules (guaranteed hours, availability, rest).

LP duals of the coverage rows are degenerate (a whole shift's cost tends to land on a single hour), so marginal
cost of demand is computed by re-solving the integer model with one more person required across a 6-hour block.

## 3. Labor demand models — `demand.py`

**Workload.** $L = \lceil T / (P\,H\,E) \rceil$ for target output $T$, units per worker-hour $P$,
hours per worker $H$ and efficiency $E$.

**Erlang C (M/M/N).** Offered load $A = \lambda \cdot \text{AHT}$. The probability of waiting is

$$
P_W(N,A) = \frac{\frac{A^N}{N!}\frac{N}{N-A}}{\sum_{k=0}^{N-1}\frac{A^k}{k!} + \frac{A^N}{N!}\frac{N}{N-A}},
\qquad
\text{SL}(N) = 1 - P_W\,e^{-(N-A)\,t^*/\text{AHT}} ,
$$

and staff on duty is the smallest $N$ with $\text{SL}(N) \ge$ target; scheduled staff is $\lceil N/(1-\text{shrinkage}) \rceil$.
Computed through the numerically stable Erlang B recursion.

**Table games.** tables $= \lceil \text{players} / (\text{seats}\times\text{occupancy}) \rceil$ clipped to
$[\text{min},\text{max}]$; dealers $= \lceil \text{tables}\times\text{relief ratio} \rceil$.
Theoretical win per table-hour $=$ hands/h $\times$ players $\times$ average bet $\times$ house edge, which ties dealer
productivity to revenue.

## 4. Forecasting — `forecast.py`

Hourly counts $y_t \sim \text{NB2}(\mu_t, \alpha)$ with

$$
\log \mu_t = \beta_0 + \beta_{\text{hour}(t)} + \gamma_{\text{dow}(t)} + \delta\,\text{FriSatEvening}_t + \eta\,\text{event}_t + \kappa\,\text{trend}_t ,
$$

$\alpha$ estimated by method of moments from a Poisson fit ($\text{Var} = \mu + \alpha\mu^2$). Rolling-origin backtest
against same-hour-last-week and 4-week same-hour average, scored on WAPE and on dealer-hours under/over-staffed.

**Staffing quantile.** Scheduling to quantile $\tau$ of the forecast is a newsvendor problem. An uncovered hour
costs $c_u = (m-1)\,w$ more than a scheduled hour; an idle scheduled hour costs $c_o = w$. The optimal quantile is

$$
\tau^* = \frac{c_u}{c_u + c_o} = \frac{m-1}{m} = \tfrac{2}{3} \text{ for } m = 3 .
$$

## 5. Experiments — `experiments.py`

* **Casino** (randomized at table-day): $\text{hands}_i = \beta_0 + \beta_1\,\text{treated}_i + \beta_2\,\text{players}_i + \text{game} + \text{dow} + \varepsilon_i$,
  standard errors clustered by table.
* **Textile** (staggered adoption): difference-in-differences with line and week fixed effects,
  $y_{lw} = \alpha_l + \lambda_w + \tau\,D_{lw} + \varepsilon_{lw}$, clustered by line; pre-period trend test for parallel trends.
  Defects use a binomial GLM on (defects, units).
* **Retail**: binomial GLM on (transactions, visitors) with store effects, clustered by store.
* Power: $n$ per arm from the two-sample t-test for a given effect and standard deviation.

## 6. Control charts — `kpi.py`

* **p-chart**, varying sample size: $\bar p \pm 3\sqrt{\bar p(1-\bar p)/n_i}$ from a baseline period, plus the
  8-points-on-one-side run rule.
* **X-mR chart**: $\hat\sigma = \overline{MR}/1.128$; rules: beyond $3\hat\sigma$, 8 on one side, 2 of 3 beyond $2\hat\sigma$.

## 7. Casino mathematics — `casino_math.py`

House edge $= -E[\text{net win per unit staked}]$; payback $= 1 -$ house edge.

* Roulette: $E = \big(c\cdot p - (N-c)\big)/N$ for $c$ numbers covered paying $p{:}1$ on an $N$-pocket wheel.
* Craps pass line: $P(\text{win}) = P(7,11) + \sum_{x\in\{4,5,6,8,9,10\}} P(x)\,\frac{P(x)}{P(x)+P(7)} = \tfrac{244}{495}$, edge $\tfrac{7}{495}$.
* Baccarat: exact enumeration of every deal from an 8-deck shoe under the standard third-card rules.
* Poker: required equity $= \text{call}/(\text{pot}+\text{call})$; $\text{EV}(\text{call}) = q\cdot\text{pot} - (1-q)\cdot\text{call}$.
