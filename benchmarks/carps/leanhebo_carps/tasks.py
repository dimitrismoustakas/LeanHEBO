"""Build CARP-S tasks from their names, using each benchmark's own search space."""

from carps.utils.task import (
    InputSpace,
    OptimizationResources,
    OutputSpace,
    Task,
    TaskMetadata,
    get_search_space_info,
)


def make_task(name: str, n_trials: int, seed: int, metric: str | None = None) -> Task:
    family, *parts = name.split("/")
    if family == "bbob":
        from carps.objective_functions.bbob import BBOBObjectiveFunction

        dimension, fid, instance = map(int, parts)
        objective = BBOBObjectiveFunction(fid, instance, dimension, seed)
        metric = "quality"
    elif family == "yahpo":
        from carps.objective_functions.yahpo import YahpoObjectiveFunction

        scenario, instance, _ = parts
        metric = metric or ("val_accuracy" if scenario == "lcbench" else "acc")
        objective = YahpoObjectiveFunction(scenario, instance, metric, seed=seed)
    elif family == "synthetic":
        from leanhebo_carps.synthetic import ConditionalQuadratic

        objective = ConditionalQuadratic(parts[0], seed)
    else:
        raise ValueError(f"Unknown task family: {family}")
    return Task(
        name=name,
        objective_function=objective,
        input_space=InputSpace(objective.configspace),
        output_space=OutputSpace(objectives=(metric or "cost",)),
        optimization_resources=OptimizationResources(n_trials=n_trials),
        metadata=TaskMetadata(**get_search_space_info(objective.configspace)),
        seed=seed,
    )
