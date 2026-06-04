from sodl_weights.pipeline import compute_pipeline_hash
from sodl_weights.semantic_router import CapabilityQuery, simple_route


def test_compute_pipeline_hash_is_stable() -> None:
    a = compute_pipeline_hash("origin:a", "semantic_chunk", {"k": 1})
    b = compute_pipeline_hash("origin:a", "semantic_chunk", {"k": 1})
    assert a == b


def test_simple_route_respects_max_results() -> None:
    query = CapabilityQuery(
        principal="carla",
        query_text="find semantic route",
        requested_caps=["semantic_search"],
        max_results=2,
    )
    result = simple_route(["a", "b", "c"], query)
    assert [item.label for item in result] == ["a", "b"]

