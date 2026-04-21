"""scenarios_profiling – scenario generation pipeline with PyTorch profiler instrumentation.

This package is a drop-in companion to ``scenarios``.  All original pipeline
logic is preserved; PyTorch ``record_function`` spans have been added to every
major stage so that a ``torch.profiler.profile`` context can attribute CPU time
(and optionally memory) to individual steps.

Quick start::

    from scenarios_profiling.generator import ScenarioGeneratorAgent
    import asyncio

    agent = ScenarioGeneratorAgent(model_id="...", show_workflow=True)

    # Run normally (profiler spans are emitted but no profiler is active –
    # they are no-ops and add negligible overhead):
    scenarios = asyncio.run(agent.run("Chiller"))

    # Run under the PyTorch profiler:
    scenarios = asyncio.run(
        agent.run_with_profiling("Chiller", profiling_dir="profiling_output/chiller")
    )

See ``profiling_utils.py`` for the ``ProfilerConfig`` dataclass and helpers.
"""
