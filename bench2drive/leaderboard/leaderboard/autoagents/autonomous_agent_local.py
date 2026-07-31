"""Compatibility alias for carla_garage-style agents ported in from fail2drive.

fail2drive's leaderboard splits its base agent into two modules -- `autonomous_agent`
(leaderboard-standard) and `autonomous_agent_local` (the variant its privileged agents
subclass, paired with `agent_wrapper_local.py` / `leaderboard_evaluator_local.py`).
This fork has only `autonomous_agent`, so `from leaderboard.autoagents import
autonomous_agent_local` -- which fail2drive's `team_code/autopilot.py` does at module
scope -- would fail here.

Re-exporting keeps those agents importable while making them subclass *this* fork's
`AutonomousAgent`, which is what we want: it carries `get_hero()` / `get_metric_info()`
and the `__init__(carla_host, carla_port, debug)` signature that this fork's
`leaderboard_evaluator.py` instantiates agents with.

The two base classes do differ in ways a ported agent may care about; a subclass that
needs them should override:
  * `set_global_plan` -- fail2drive's stores the undownsampled route as
    `org_dense_route_gps` / `org_dense_route_world_coord` (PrivilegedRoutePlanner needs
    it) and downsamples the sparse plan by 200, not 50.
  * `__call__` / `run_step` -- fail2drive's passes a `sensors` argument through.
See team_code/pdm_lite_b2d_agent.py for a worked example.
"""

from leaderboard.autoagents.autonomous_agent import AutonomousAgent, Track  # noqa: F401
