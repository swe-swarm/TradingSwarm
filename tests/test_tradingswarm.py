from unittest.mock import patch

from typer.testing import CliRunner

from cli.main import app
from tradingagents.default_config import _apply_env_overrides, config_env


def test_public_graph_and_config_are_legacy_objects():
    from tradingagents.default_config import DEFAULT_CONFIG
    from tradingagents.graph.trading_graph import TradingAgentsGraph
    from tradingswarm import DEFAULT_CONFIG as swarm_config, TradingSwarmGraph

    assert TradingSwarmGraph is TradingAgentsGraph
    assert swarm_config is DEFAULT_CONFIG


def test_current_environment_wins(monkeypatch):
    monkeypatch.setenv("TRADINGAGENTS_MAX_DEBATE_ROUNDS", "broken")
    monkeypatch.setenv("TRADINGSWARM_MAX_DEBATE_ROUNDS", "3")
    assert _apply_env_overrides({"max_debate_rounds": 1})["max_debate_rounds"] == 3


def test_empty_current_environment_falls_back(monkeypatch):
    monkeypatch.setenv("TRADINGAGENTS_LLM_PROVIDER", "copilot")
    monkeypatch.setenv("TRADINGSWARM_LLM_PROVIDER", "")
    assert config_env("TRADINGAGENTS_LLM_PROVIDER") == "copilot"


def test_cli_exposes_rebrand_and_protocol_commands():
    result = CliRunner().invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "TradingSwarm" in result.output
    assert "fleet" in result.output
    assert "acp" in result.output


def test_acp_command_dispatches_without_interactive_prompts():
    import sys
    from types import SimpleNamespace
    from unittest.mock import Mock

    main = Mock()
    with patch.dict(sys.modules, {"tradingswarm.acp": SimpleNamespace(main=main)}):
        result = CliRunner().invoke(app, ["acp"])
    assert result.exit_code == 0
    main.assert_called_once_with()


def test_announcements_do_not_contact_upstream():
    from cli.announcements import fetch_announcements

    with patch("cli.announcements.requests.get") as get:
        result = fetch_announcements()
    get.assert_not_called()
    assert "swe-swarm/TradingSwarm" in result["announcements"][0]
