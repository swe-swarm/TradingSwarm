"""TradingSwarm public API, backed by the compatible TradingAgents engine."""

__all__ = ["DEFAULT_CONFIG", "TradingSwarmGraph"]


def __getattr__(name):
    if name == "DEFAULT_CONFIG":
        from tradingagents.default_config import DEFAULT_CONFIG

        return DEFAULT_CONFIG
    if name == "TradingSwarmGraph":
        from tradingagents.graph.trading_graph import TradingAgentsGraph

        return TradingAgentsGraph
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
