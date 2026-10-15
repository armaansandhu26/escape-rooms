from motiverooms.llm_coord import decision_order


def test_decision_order_alternates_first_agent() -> None:
    assert decision_order(("A", "B"), 0) == ("A", "B")
    assert decision_order(("A", "B"), 1) == ("B", "A")
