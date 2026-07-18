"""
cortex.py — The Executive Brain of Genesis.

WHY this exists
───────────────
Phase 1 and 2 built independent cognitive modules (RND, D1, World Model,
contrastive learning, etc.) but left a critical gap: there was no system
deciding HOW to learn. Curiosity algorithm, replay strategy, exploration
schedule — all were hardcoded by the human researcher. A biological organism
does not have a researcher twisting its curiosity knob. It regulates itself.

The Executive Cortex is the first meta-cognitive system in Genesis. It does
NOT interact with the environment, process pixels, or perform world modelling.
Instead, it OBSERVES Genesis's own internal dynamics and ADAPTS learning
parameters continuously.

Cognitive parallel: the prefrontal cortex's metacognitive monitoring — the
ability to reflect on one's own thinking and adjust learning strategies
accordingly (cf. "thinking about thinking," Flavell 1979; "cognitive control,"
Miller & Cohen 2001).

What it regulates
─────────────────
  • Curiosity weights — which intrinsic signal to trust, and by how much
  • Memory mixing — how to sample experience (uniform vs prioritized replay)
  • Exploration — when to explore versus exploit
  • Learning rates — how fast each subsystem should learn

Design rule
───────────
The Executive Cortex NEVER uses hardcoded rules ("if maze: use ICM").
It observes internal dynamics and adapts automatically through continuous
feedback.

Future expansion
────────────────
Every future cognitive subsystem (Attention, Planning, Language, Reasoning,
Emotion, Self Model) should communicate through the Executive Cortex.
It is designed to become the permanent central nervous system of Genesis.
"""
import numpy as np
from collections import deque


class MetricBuffer:
    """Rolling-window metric storage with trend computation.

    WHY: Detecting whether a metric is improving, stagnating, or degrading
    requires tracking recent history, not just the latest value. This is the
    sensory epithelium of the Executive Cortex — what it "sees" of Genesis.

    Ponytail: linear trend via endpoint difference. A proper linear regression
    would be more robust to noise; endpoint diff is O(1) and sufficient when
    the regulation interval is already a smoothed average.
    """
    def __init__(self, maxlen=200):
        self.maxlen = maxlen
        self.buf = deque(maxlen=maxlen)

    def append(self, value):
        self.buf.append(float(value))

    def __len__(self):
        return len(self.buf)

    def mean(self, n=None):
        if not self.buf:
            return 0.0
        data = list(self.buf)
        if n is not None:
            data = data[-min(n, len(data)):]
        return float(np.mean(data))

    def std(self, n=None):
        if len(self.buf) < 2:
            return 1.0
        data = list(self.buf)
        if n is not None:
            data = data[-min(n, len(data)):]
        return float(np.std(data)) + 1e-8

    def trend(self, n=50):
        """Endpoint trend: positive = increasing, negative = decreasing.
        ponytail: linear regression would be more accurate, endpoint diff
        is O(1) and works for regulation where only direction matters.
        """
        if len(self.buf) < n + 1:
            return 0.0
        data = list(self.buf)
        recent = data[-n:]
        base = float(recent[0])
        if abs(base) < 1e-8:
            return float(recent[-1] - base)
        return (recent[-1] - base) / max(1e-8, abs(base))

    def normalize(self, value):
        """Z-score normalize a value against history."""
        return (value - self.mean()) / self.std()

    def recent(self, n=50):
        data = list(self.buf)
        return data[-min(n, len(data)):]


class ExecutiveCortex:
    """The Executive Brain — observes Genesis and regulates learning adaptively.

    Usage:
        cortex = ExecutiveCortex()

        # Inside training loop, after each update:
        cortex.observe(
            global_step=step,
            coverage=env.coverage(),
            rnd_reward=rnd_reward,
            icm_reward=icm_reward,
            td_error=stats['td_error_mean'],
            wm_loss=wm_loss,
            policy_loss=policy_loss,
            representation_variance=var,
            ...
        )

        # Periodically:
        params = cortex.regulate()
        # params: {curiosity_weights, memory_weights, epsilon, lr_factors}

    NEVER hardcodes rules. All regulation is driven by observed metric trends.
    """
    def __init__(self, config=None):
        cfg = {
            'window': 200,
            'regulate_every': 10,
            'curiosity_alpha': 0.1,
            'curiosity_temp': 1.0,
            'curiosity_min_weight': 0.05,
            'memory_alpha': 0.1,
            'memory_temp': 0.5,
            'exploration_alpha': 0.05,
            'exploration_min': 0.01,
            'exploration_max': 1.0,
            'lr_alpha': 0.05,
            'lr_min_factor': 0.5,
            'lr_max_factor': 2.0,
            'goal_alpha': 0.1,
            'goal_temp': 0.5,
            'meta_window': 500,
        }
        if config:
            cfg.update(config)
        self.cfg = cfg

        self.buf = {}
        self._ensure('_regulate_count')

        self.global_step = 0
        self.episode = 0

        self._curiosity_w = {'rnd': 1.0, 'icm': 1.0}
        self._memory_w = {'uniform': 1.0, 'prioritized': 1.0, 'sequence': 1.0}
        self._epsilon = 1.0
        self._lr_factors = {}
        self._goal_weights = {
            'curiosity': 1.0, 'safety': 0.2, 'efficiency': 0.2,
            'knowledge': 1.0, 'survival': 0.2, 'exploration': 1.0, 'prediction': 1.0,
        }

        self._meta_history = deque(maxlen=cfg['meta_window'])

    def _ensure(self, name, maxlen=None):
        if maxlen is None:
            maxlen = self.cfg['window']
        if name not in self.buf:
            self.buf[name] = MetricBuffer(maxlen)

    def observe(self, **metrics):
        """Feed metrics from all subsystems.

        Call this every training step with whatever metrics are available.
        Missing metrics are silently ignored; the cortex adapts based on
        whatever signals it has.
        """
        self.global_step = metrics.get('global_step', self.global_step + 1)
        self.episode = metrics.get('episode', self.episode)

        for key, value in metrics.items():
            if key in ('global_step', 'episode'):
                continue
            if isinstance(value, (int, float, np.floating)):
                self._ensure(key)
                self.buf[key].append(float(value))

    def _read(self, name, default=0.0):
        buf = self.buf.get(name)
        if buf is None or len(buf) < 2:
            return default
        return buf

    # ── Curiosity Regulation ──────────────────────────────────────────────

    def regulate_curiosity(self):
        """Weight each curiosity module by its recent reward magnitude.

        Principle: a module producing large intrinsic rewards is finding
        novelty the agent hasn't mastered. Weight it more. A module producing
        small/noisy rewards has saturated — weight it less.

        This naturally handles the RND-vs-ICM tradeoff: early exploration,
        RND's pure novelty signal dominates; once the agent has seen most
        states, RND saturates and ICM's action-conditional prediction error
        takes over. No hardcoded switch.

        ponytail: softmax on mean reward magnitude. A more sophisticated
        approach could use uncertainty- or information-gain-weighted mixing.
        """
        rewards = {}
        for mod in self._curiosity_w:
            buf = self._read(f'{mod}_reward', 0.0)
            m = abs(buf.mean(50)) if isinstance(buf, MetricBuffer) else 0.0
            rewards[mod] = m + self.cfg['curiosity_min_weight']  # floor

        total = sum(rewards.values())
        if total > 0:
            temp = self.cfg['curiosity_temp']
            keys = list(rewards.keys())
            vals = np.array([rewards[k] / temp for k in keys])
            vals = np.exp(vals - vals.max())
            target = vals / (vals.sum() + 1e-8)
            alpha = self.cfg['curiosity_alpha']
            for i, k in enumerate(keys):
                self._curiosity_w[k] = (1 - alpha) * self._curiosity_w[k] + alpha * float(target[i])

        total = sum(self._curiosity_w.values())
        if total > 0:
            for k in self._curiosity_w:
                self._curiosity_w[k] /= total

        return dict(self._curiosity_w)

    # ── Memory Regulation ─────────────────────────────────────────────────

    def regulate_memory(self):
        """Weight replay types by learning improvement (TD error trend).

        Principle: a replay type whose TD error is decreasing (negative trend)
        is surfacing transitions the agent is actively learning from — weight
        it more. A type with increasing or flat TD error has saturated or
        is sampling unhelpful noise — weight it less.

        Trend-based avoids the positive-feedback trap of absolute-magnitude
        regulation: prioritizing high-error transitions keeps the measured
        TD error artificially high, which would otherwise reinforce the same
        type regardless of actual learning progress.
        """
        errors = {}
        for mem in self._memory_w:
            buf = self._read(f'{mem}_td_error', 0.0)
            if isinstance(buf, MetricBuffer) and len(buf) >= 10:
                t = buf.trend(30)
                # negative trend = improving → weight goes up
                # positive trend = regressing → weight goes down
                errors[mem] = float(np.clip(-t, 0.0, None)) + 1e-8
            elif isinstance(buf, MetricBuffer):
                errors[mem] = 1.0  # neutral before enough data
            else:
                errors[mem] = 1e-8  # untracked type, effectively zero

        total = sum(errors.values())
        if total > 0:
            temp = self.cfg['memory_temp']
            keys = list(errors.keys())
            vals = np.array([errors[k] / temp for k in keys])
            vals = np.exp(vals - vals.max())
            target = vals / (vals.sum() + 1e-8)
            alpha = self.cfg['memory_alpha']
            for i, k in enumerate(keys):
                self._memory_w[k] = (1 - alpha) * self._memory_w[k] + alpha * float(target[i])

        total = sum(self._memory_w.values())
        if total > 0:
            for k in self._memory_w:
                self._memory_w[k] /= total

        return dict(self._memory_w)

    # ── Exploration Regulation ────────────────────────────────────────────

    def regulate_exploration(self):
        """Adjust exploration rate based on coverage improvement.

        Principle: when coverage is improving, the agent is already exploring
        effectively — reduce epsilon to let it exploit what it's found. When
        coverage stagnates, increase exploration to escape the local area.

        This is a negative-feedback integral controller on coverage trend.
        No hardcoded schedule — adapts to how fast the agent is actually
        discovering new states.

        ponytail: single coverage metric. Future versions should track
        per-region coverage to avoid the "95% saturated but still exploring"
        edge case.
        """
        buf = self._read('coverage', 0.0)
        if not isinstance(buf, MetricBuffer):
            return self._epsilon

        trend = buf.trend(50)
        adjustment = -trend * self.cfg['exploration_alpha']

        self._epsilon = float(np.clip(
            self._epsilon + adjustment,
            self.cfg['exploration_min'],
            self.cfg['exploration_max'],
        ))

        return self._epsilon

    # ── Learning Rate Regulation ──────────────────────────────────────────

    def regulate_learning_rates(self, available_modules=None):
        """Adjust learning rate factors based on loss trends.

        Principle: if a module's loss is decreasing steadily, it's learning
        well — can increase its learning rate. If loss is oscillating or
        increasing, decrease learning rate to stabilize.

        ponytail: one LR factor per tracked loss, modulated by trend sign.
        A full implementation would use per-parameter adaptive rates and
        gradient-noise-aware scheduling.
        """
        if available_modules is None:
            available_modules = ['policy_loss', 'wm_loss', 'rnd_loss', 'contrastive_loss']

        result = {}
        for module in available_modules:
            buf = self._read(module, 0.0)
            if not isinstance(buf, MetricBuffer):
                continue

            trend = buf.trend(30)

            if module not in self._lr_factors:
                self._lr_factors[module] = 1.0

            adjustment = -trend * self.cfg['lr_alpha']
            self._lr_factors[module] = float(np.clip(
                self._lr_factors[module] + adjustment,
                self.cfg['lr_min_factor'],
                self.cfg['lr_max_factor'],
            ))
            result[module] = self._lr_factors[module]

        return result

    # ── Goal Regulation ──────────────────────────────────────────────────

    def regulate_goals(self):
        """Balance multiple motivational drives based on internal state.

        Each goal's weight is adjusted by:
          - curiosity: stronger when coverage is low (novelty needed)
          - safety: stronger when negative outcomes detected
          - efficiency: stronger when coverage is high (exploit mode)
          - knowledge: stronger when WM uncertainty is high
          - exploration: stronger when coverage stagnates
          - prediction: stronger when WM loss is high

        ponytail: independent adjustment per goal. Competing goals would
        need a proper multi-objective optimization in the full version.
        """
        coverage_buf = self._read('coverage', 0.0)
        coverage = coverage_buf.mean(50) if isinstance(coverage_buf, MetricBuffer) else 0.0
        coverage_trend = coverage_buf.trend(50) if isinstance(coverage_buf, MetricBuffer) else 0.0

        wm_buf = self._read('wm_loss', 0.0)
        wm_loss = wm_buf.mean(50) if isinstance(wm_buf, MetricBuffer) else 0.0

        td_buf = self._read('td_error_mean', 0.0)
        td_error = td_buf.mean(50) if isinstance(td_buf, MetricBuffer) else 0.0

        adjustments = {}

        adjustments['curiosity'] = 1.0 - coverage  # explore when coverage low
        adjustments['exploration'] = 1.0 if coverage_trend < 0.01 else 0.3
        adjustments['knowledge'] = float(np.clip(wm_loss * 10, 0.0, 2.0))
        adjustments['prediction'] = float(np.clip(wm_loss * 5, 0.0, 2.0))
        adjustments['efficiency'] = coverage  # exploit when coverage high
        adjustments['safety'] = 0.2 + float(np.clip(td_error * 5, 0.0, 0.8))
        adjustments['survival'] = 0.2

        alpha = self.cfg['goal_alpha']
        for key, target in adjustments.items():
            if key in self._goal_weights:
                self._goal_weights[key] = (1 - alpha) * self._goal_weights[key] + alpha * target

        total = sum(self._goal_weights.values())
        if total > 0:
            for k in self._goal_weights:
                self._goal_weights[k] /= total

        return dict(self._goal_weights)

    @property
    def goal_weights(self):
        return dict(self._goal_weights)

    # ── Main Regulation Loop ──────────────────────────────────────────────

    def regulate(self):
        """Run one regulation cycle. Returns adaptive parameters dict.

        Only runs full regulation every `regulate_every` steps to avoid
        oscillation. Between cycles, the previous parameters remain in effect.

        Returns
        -------
        dict with keys:
            curiosity_weights : {'rnd': w, 'icm': w}
            memory_weights   : {'uniform': w, 'prioritized': w, 'sequence': w}
            epsilon          : float
            lr_factors       : {loss_name: factor}
        """
        params = {}

        regulate_count = self.buf.get('_regulate_count', MetricBuffer(1))
        do_regulate = (self.global_step - (regulate_count.mean() if len(regulate_count) > 0 else 0)
                       >= self.cfg['regulate_every'])

        if do_regulate and self.global_step > 0:
            self.buf['_regulate_count'] = MetricBuffer(1)
            self.buf['_regulate_count'].append(self.global_step)

            params['curiosity_weights'] = self.regulate_curiosity()
            params['memory_weights'] = self.regulate_memory()
            params['epsilon'] = self.regulate_exploration()
            params['lr_factors'] = self.regulate_learning_rates()
            params['goal_weights'] = self.regulate_goals()
            params['n_goals'] = len(self._goal_weights)

            self._meta_history.append({
                'step': self.global_step,
                'params': {
                    'curiosity_weights': dict(self._curiosity_w),
                    'memory_weights': dict(self._memory_w),
                    'epsilon': self._epsilon,
                    'lr_factors': dict(self._lr_factors),
                    'goal_weights': dict(self._goal_weights),
                },
                'metrics': {
                    k: buf.mean(10) if isinstance(buf := self.buf.get(k), MetricBuffer) else 0.0
                    for k in ('td_error_mean', 'wm_loss', 'coverage', 'intrinsic_return')
                    if k in self.buf
                },
            })

        return params

    # ── Public Accessors ──────────────────────────────────────────────────

    @property
    def curiosity_weights(self):
        return dict(self._curiosity_w)

    @property
    def memory_weights(self):
        return dict(self._memory_w)

    @property
    def epsilon(self):
        return self._epsilon

    @property
    def lr_factors(self):
        return dict(self._lr_factors)

    def get_state(self):
        """Full snapshot for logging/diagnostics."""
        return {
            'step': self.global_step,
            'episode': self.episode,
            'curiosity_weights': dict(self._curiosity_w),
            'memory_weights': dict(self._memory_w),
            'epsilon': self._epsilon,
            'lr_factors': dict(self._lr_factors),
            'goal_weights': dict(self._goal_weights),
            'meta_history_length': len(self._meta_history),
        }

    def get_summary(self, metrics=None):
        """Human-readable summary of current regulation state."""
        if metrics is None:
            metrics = ['td_error_mean', 'wm_loss', 'rnd_reward', 'icm_reward', 'coverage']
        summary = {'state': self.get_state(), 'metric_buffers': {}}
        for m in metrics:
            buf = self.buf.get(m)
            if buf is not None:
                summary['metric_buffers'][m] = {
                    'mean': buf.mean(50),
                    'std': buf.std(50),
                    'trend': buf.trend(50),
                }
        return summary
