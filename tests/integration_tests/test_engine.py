import pytest

from graphgen.engine import Context, Engine, op

engine = Engine(max_workers=2)


def test_simple_dag(capsys):
    """Verify the DAG A->B/C->D execution results and print order."""
    ctx = Context()

    @op("A")
    def op_a(self, ctx):
        print("Running A")
        ctx.set("A", 1)

    @op("B", deps=["A"])
    def op_b(self, ctx):
        print("Running B")
        ctx.set("B", ctx.get("A") + 1)

    @op("C", deps=["A"])
    def op_c(self, ctx):
        print("Running C")
        ctx.set("C", ctx.get("A") + 2)

    @op("D", deps=["B", "C"])
    def op_d(self, ctx):
        print("Running D")
        ctx.set("D", ctx.get("B") + ctx.get("C"))

    # Explicitly list the nodes to run; avoid relying on globals().
    ops = [op_a, op_b, op_c, op_d]
    engine.run(ops, ctx)

    # Assert final results.
    assert ctx["A"] == 1
    assert ctx["B"] == 2
    assert ctx["C"] == 3
    assert ctx["D"] == 5

    # Assert print order: A must run before B and C; D must run after B and C.
    captured = capsys.readouterr().out.strip().splitlines()
    assert "Running A" in captured
    assert "Running B" in captured
    assert "Running C" in captured
    assert "Running D" in captured

    a_idx = next(i for i, line in enumerate(captured) if "Running A" in line)
    b_idx = next(i for i, line in enumerate(captured) if "Running B" in line)
    c_idx = next(i for i, line in enumerate(captured) if "Running C" in line)
    d_idx = next(i for i, line in enumerate(captured) if "Running D" in line)

    assert a_idx < b_idx
    assert a_idx < c_idx
    assert d_idx > b_idx
    assert d_idx > c_idx


def test_cyclic_detection():
    """A cyclic dependency should raise ValueError."""
    ctx = Context()

    @op("X", deps=["Y"])
    def op_x(self, ctx):
        pass

    @op("Y", deps=["X"])
    def op_y(self, ctx):
        pass

    ops = [op_x, op_y]
    with pytest.raises(ValueError, match="Cyclic dependencies"):
        engine.run(ops, ctx)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
