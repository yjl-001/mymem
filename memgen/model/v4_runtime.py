"""Gate-scoped memory-episode state machine for MemGen V4.

This module intentionally contains no model forward pass.  It defines the
auditable lifecycle that a GPU runtime must obey:

* gate and selector run only while ARMED and no memory is active;
* selection activates one target bank for a local episode;
* two consecutive low-entropy observations end a recovered episode;
* direct side-KV visibility is capped at thirty-two decode steps;
* abstention consumes an attempt but is not terminal;
* every question receives at most three selector attempts.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from memgen.experience.v4_bank import (
    V4_MAX_ACTIVE_STEPS,
    V4_MAX_SELECTOR_ATTEMPTS,
    V4_RECOVERY_LOW_TOKEN_COUNT,
)


V4_EPISODE_CONFIG_SCHEMA = "memgen-v4-episode-config-v1"
V4_EPISODE_TRANSITION_SCHEMA = "memgen-v4-episode-transition-v1"
V4_LIFECYCLE_STATES = frozenset(
    {"ARMED", "ACTIVE", "COOLDOWN", "EXHAUSTED", "CLOSED"}
)


@dataclass(frozen=True)
class V4EpisodeConfig:
    max_selector_attempts: int = V4_MAX_SELECTOR_ATTEMPTS
    recovery_low_token_count: int = V4_RECOVERY_LOW_TOKEN_COUNT
    max_active_steps: int = V4_MAX_ACTIVE_STEPS
    abstain_policy: str = "consume_attempt_then_cooldown_nonterminal"
    active_selection_policy: str = "no_reselection_within_episode"
    reselect_same_bank_after_recovery: bool = True
    schema_version: str = V4_EPISODE_CONFIG_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != V4_EPISODE_CONFIG_SCHEMA:
            raise ValueError("Unexpected V4 episode-config schema")
        if self.max_selector_attempts != 3:
            raise ValueError("V4 initial runtime allows exactly three selector attempts")
        if self.recovery_low_token_count != 2:
            raise ValueError("V4 recovery requires exactly two low-entropy tokens")
        if self.max_active_steps != 32:
            raise ValueError("V4 direct side-KV visibility is capped at thirty-two steps")
        if self.abstain_policy != "consume_attempt_then_cooldown_nonterminal":
            raise ValueError("Unexpected V4 abstain policy")
        if self.active_selection_policy != "no_reselection_within_episode":
            raise ValueError("Unexpected V4 active selection policy")
        if self.reselect_same_bank_after_recovery is not True:
            raise ValueError("V4 permits the same bank in a later recovered episode")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class V4EpisodeTransition:
    event: str
    state_before: str
    state_after: str
    attempt_count: int
    active_bank_before: str | None
    active_bank_after: str | None
    active_step_count_before: int
    active_step_count_after: int
    low_entropy_streak_before: int
    low_entropy_streak_after: int
    deactivate_memory: bool
    activation_bank_id: str | None
    reason: str
    schema_version: str = V4_EPISODE_TRANSITION_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != V4_EPISODE_TRANSITION_SCHEMA:
            raise ValueError("Unexpected V4 episode-transition schema")
        if self.state_before not in V4_LIFECYCLE_STATES:
            raise ValueError("Unexpected V4 state_before")
        if self.state_after not in V4_LIFECYCLE_STATES:
            raise ValueError("Unexpected V4 state_after")
        if self.deactivate_memory and self.active_bank_before is None:
            raise ValueError("V4 transition cannot deactivate absent memory")
        if self.activation_bank_id is not None and self.active_bank_after != self.activation_bank_id:
            raise ValueError("V4 transition activation identity mismatch")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class V4MemoryEpisodeController:
    """Deterministic lifecycle controller, independent of model implementation."""

    def __init__(self, config: V4EpisodeConfig | None = None) -> None:
        self.config = config or V4EpisodeConfig()
        self.state = "ARMED"
        self.attempt_count = 0
        self.active_bank_id: str | None = None
        self.active_step_count = 0
        self.low_entropy_streak = 0
        self.transitions: list[V4EpisodeTransition] = []

    @property
    def gate_enabled(self) -> bool:
        return self.state == "ARMED" and self.active_bank_id is None

    @property
    def memory_should_be_active(self) -> bool:
        return self.state == "ACTIVE" and self.active_bank_id is not None

    def _record(
        self,
        *,
        event: str,
        state_before: str,
        active_bank_before: str | None,
        active_step_count_before: int,
        low_entropy_streak_before: int,
        deactivate_memory: bool,
        activation_bank_id: str | None,
        reason: str,
    ) -> V4EpisodeTransition:
        transition = V4EpisodeTransition(
            event=event,
            state_before=state_before,
            state_after=self.state,
            attempt_count=self.attempt_count,
            active_bank_before=active_bank_before,
            active_bank_after=self.active_bank_id,
            active_step_count_before=active_step_count_before,
            active_step_count_after=self.active_step_count,
            low_entropy_streak_before=low_entropy_streak_before,
            low_entropy_streak_after=self.low_entropy_streak,
            deactivate_memory=deactivate_memory,
            activation_bank_id=activation_bank_id,
            reason=reason,
        )
        self.transitions.append(transition)
        return transition

    def apply_selection(self, selected_bank_id: str | None) -> V4EpisodeTransition:
        """Consume one positive-gate selector decision while ARMED."""

        if not self.gate_enabled:
            raise RuntimeError("V4 selection is allowed only while ARMED")
        state_before = self.state
        active_before = self.active_bank_id
        active_step_count_before = self.active_step_count
        low_entropy_streak_before = self.low_entropy_streak
        self.attempt_count += 1
        if self.attempt_count > self.config.max_selector_attempts:
            raise RuntimeError("V4 selector attempt budget was exceeded")
        self.active_step_count = 0
        self.low_entropy_streak = 0
        if selected_bank_id is None:
            self.state = (
                "EXHAUSTED"
                if self.attempt_count >= self.config.max_selector_attempts
                else "COOLDOWN"
            )
            return self._record(
                event="selector_abstained",
                state_before=state_before,
                active_bank_before=active_before,
                active_step_count_before=active_step_count_before,
                low_entropy_streak_before=low_entropy_streak_before,
                deactivate_memory=False,
                activation_bank_id=None,
                reason="abstain_consumed_attempt",
            )
        if not isinstance(selected_bank_id, str) or not selected_bank_id:
            raise ValueError("V4 selected bank ID must be non-empty")
        self.active_bank_id = selected_bank_id
        self.state = "ACTIVE"
        return self._record(
            event="selector_selected",
            state_before=state_before,
            active_bank_before=active_before,
            active_step_count_before=active_step_count_before,
            low_entropy_streak_before=low_entropy_streak_before,
            deactivate_memory=False,
            activation_bank_id=selected_bank_id,
            reason="target_episode_started",
        )

    def observe_decoded_token(
        self,
        *,
        low_entropy: bool,
        answer_marker_seen: bool = False,
        eos_seen: bool = False,
    ) -> V4EpisodeTransition | None:
        """Advance recovery/window accounting after one actual-path token."""

        if self.state in {"CLOSED", "EXHAUSTED"}:
            if answer_marker_seen or eos_seen:
                return self.close(reason="answer_or_eos")
            return None
        if answer_marker_seen or eos_seen:
            return self.close(reason="answer_or_eos")
        if self.state == "ARMED":
            return None
        state_before = self.state
        active_before = self.active_bank_id
        active_step_count_before = self.active_step_count
        low_entropy_streak_before = self.low_entropy_streak
        if self.state == "ACTIVE":
            self.active_step_count += 1
        self.low_entropy_streak = (
            self.low_entropy_streak + 1 if low_entropy else 0
        )

        if self.state == "ACTIVE" and (
            self.low_entropy_streak >= self.config.recovery_low_token_count
        ):
            self.active_bank_id = None
            self.active_step_count = 0
            self.low_entropy_streak = 0
            self.state = (
                "EXHAUSTED"
                if self.attempt_count >= self.config.max_selector_attempts
                else "ARMED"
            )
            return self._record(
                event="memory_deactivated",
                state_before=state_before,
                active_bank_before=active_before,
                active_step_count_before=active_step_count_before,
                low_entropy_streak_before=low_entropy_streak_before,
                deactivate_memory=True,
                activation_bank_id=None,
                reason="recovery_low_entropy_hysteresis",
            )

        if self.state == "ACTIVE" and (
            self.active_step_count >= self.config.max_active_steps
        ):
            self.active_bank_id = None
            self.active_step_count = 0
            self.low_entropy_streak = 0
            self.state = (
                "EXHAUSTED"
                if self.attempt_count >= self.config.max_selector_attempts
                else "COOLDOWN"
            )
            return self._record(
                event="memory_deactivated",
                state_before=state_before,
                active_bank_before=active_before,
                active_step_count_before=active_step_count_before,
                low_entropy_streak_before=low_entropy_streak_before,
                deactivate_memory=True,
                activation_bank_id=None,
                reason="maximum_active_window",
            )

        if self.state == "COOLDOWN" and (
            self.low_entropy_streak >= self.config.recovery_low_token_count
        ):
            self.low_entropy_streak = 0
            self.state = (
                "EXHAUSTED"
                if self.attempt_count >= self.config.max_selector_attempts
                else "ARMED"
            )
            return self._record(
                event="gate_rearmed",
                state_before=state_before,
                active_bank_before=active_before,
                active_step_count_before=active_step_count_before,
                low_entropy_streak_before=low_entropy_streak_before,
                deactivate_memory=False,
                activation_bank_id=None,
                reason="cooldown_low_entropy_hysteresis",
            )
        return None

    def close(self, *, reason: str) -> V4EpisodeTransition:
        if self.state == "CLOSED":
            return self.transitions[-1]
        state_before = self.state
        active_before = self.active_bank_id
        active_step_count_before = self.active_step_count
        low_entropy_streak_before = self.low_entropy_streak
        deactivate = self.active_bank_id is not None
        self.active_bank_id = None
        self.active_step_count = 0
        self.low_entropy_streak = 0
        self.state = "CLOSED"
        return self._record(
            event="closed",
            state_before=state_before,
            active_bank_before=active_before,
            active_step_count_before=active_step_count_before,
            low_entropy_streak_before=low_entropy_streak_before,
            deactivate_memory=deactivate,
            activation_bank_id=None,
            reason=reason,
        )

    def summary(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "attempt_count": self.attempt_count,
            "active_bank_id": self.active_bank_id,
            "active_step_count": self.active_step_count,
            "low_entropy_streak": self.low_entropy_streak,
            "gate_enabled": self.gate_enabled,
            "memory_should_be_active": self.memory_should_be_active,
            "transition_count": len(self.transitions),
            "transitions": [item.to_dict() for item in self.transitions],
            "config": self.config.to_dict(),
        }


__all__ = [
    "V4EpisodeConfig",
    "V4EpisodeTransition",
    "V4MemoryEpisodeController",
]
