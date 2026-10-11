from src.registry.quality_gate import run_quality_gate


def test_quality_gate_executes_champion_vs_challenger():
    """Verify that the quality gate successfully evaluates candidate vs production."""
    result = run_quality_gate(min_improvement=0.0)

    assert result["decision"] in ["PROMOTED", "REJECTED", "NO_OP"]
    assert "production_version" in result
    if result["decision"] in ["PROMOTED", "REJECTED"]:
        assert "candidate_version" in result
        assert 0.0 <= result["production_pr_auc"] <= 1.0
        assert 0.0 <= result["candidate_pr_auc"] <= 1.0
