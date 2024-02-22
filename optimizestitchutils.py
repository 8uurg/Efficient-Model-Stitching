def problem_constructor(problem_name, dataset_path):
    import recomb.problems as problems
    if problem_name == "imagenet":
        return problems.ImageNetProblem(root=dataset_path, validation_sample_limit=1000)
    elif problem_name == "voc":
        return problems.VOCSegmentationProblem(root=dataset_path, batched_validation=True)
    else:
        raise ValueError(f"Unknown problem {problem_name}")

def noop(_th):
    pass

def get_approach(async_loop, approach_spec, problem_spec):
    import ealib
    import math

    eq = ealib.PythonAsyncIOEQ(async_loop)
    scheduler = ealib.Scheduler(eq)
    do_adaptive_steering = approach_spec.get("adaptive_steering", False)
    init_p = approach_spec.get("init_p", 0.1)
    if "GOMEA" in approach_spec["name"]:
        objective_indices = [0, 1]
        # Note - general assumption on the genotype used here.
        initializer = ealib.CategoricalProbabilisticallyCompleteInitializer(
            [[1.0 - init_p, init_p] if a == 2 else [1.0, 1.0, 1.0]
            for a in problem_spec["alphabet_size"]]
        )
        fos_learner = ealib.CategoricalLinkageTree(
            ealib.NMI(),
            ealib.FoSOrdering.Random,
            prune_root=True,
        )
        acceptance_criterion = ealib.ScalarizationAcceptanceCriterion(
            ealib.TschebysheffObjectiveScalarizer(objective_indices)
        )
        archive = ealib.BruteforceArchive(objective_indices)

        steadystate = approach_spec["name"].endswith("-SS")

        if do_adaptive_steering:
            # Add additional acceptance_criterion for thresholding.
            th_a_c = ealib.ThresholdAcceptanceCriterion(0, math.inf, True)
            acceptance_criterion = ealib.SequentialCombineAcceptanceCriterion([th_a_c, acceptance_criterion])

            def set_adaptive_steering_threshold(threshold):
                th_a_c.set_threshold(threshold)
                archive.set_threshold(0, threshold)
        else:
            set_adaptive_steering_threshold = noop

        if approach_spec["name"].startswith("LK-GOMEA"):
            return scheduler, ealib.DistributedAsynchronousKernelGOMEA(
                scheduler,
                approach_spec["population_size"],
                approach_spec["num_clusters"],
                objective_indices,
                initializer,
                fos_learner,
                acceptance_criterion,
                archive,
                steadystate=steadystate,
                lkg_strategy=ealib.SimpleStrategy(ealib.LKSimpleStrategy.RAND_ASYM) ), set_adaptive_steering_threshold
        else:
            return scheduler, ealib.DistributedAsynchronousGOMEA(
                scheduler,
                approach_spec["population_size"],
                approach_spec["num_clusters"],
                objective_indices,
                initializer,
                fos_learner,
                acceptance_criterion,
                archive,
                steadystate=steadystate), set_adaptive_steering_threshold
    if approach_spec["name"] == "SGA" or approach_spec["name"] == "SGA-1PX":
        objective_indices = [0, 1]
        # Note - general assumption on the genotype used here.
        initializer = ealib.CategoricalProbabilisticallyCompleteInitializer(
            [[1 - init_p, init_p] if a == 2 else [1.0, 1.0, 1.0]
            for a in problem_spec["alphabet_size"]]
        )
        crossover = ealib.KPointCrossover(2 if approach_spec["name"] != "SGA-1PX" else 1)
        mutation = ealib.PerVariableInAlphabetMutation(1 / len(problem_spec["alphabet_size"]))
        parent_selection = ealib.ShuffledSequentialSelection()

        acceptance_criterion = ealib.ScalarizationAcceptanceCriterion(
            ealib.TschebysheffObjectiveScalarizer(objective_indices),
            # Offspring do NOT necessarily have weights set, and should compete against their parent on equal footing.
            True
        )
        archive = ealib.BruteforceArchive(objective_indices)
        # Familial replacement
        replacement_strategy = 7

        if do_adaptive_steering:
            # Add additional acceptance_criterion for thresholding.
            th_a_c = ealib.ThresholdAcceptanceCriterion(0, math.inf, True)
            acceptance_criterion = ealib.SequentialCombineAcceptanceCriterion([th_a_c, acceptance_criterion])

            def set_adaptive_steering_threshold(threshold):
                th_a_c.set_threshold(threshold)
                archive.set_threshold(0, threshold)
        else:
            set_adaptive_steering_threshold = noop

        return scheduler, ealib.DistributedAsynchronousSimpleGA(
            scheduler,
            approach_spec["population_size"],
            approach_spec["population_size"],
            replacement_strategy,
            initializer,
            crossover,
            mutation,
            parent_selection,
            acceptance_criterion,
            archive
        ), set_adaptive_steering_threshold
    if approach_spec["name"] == "RS":
        objective_indices = [0, 1]
        # Note - general assumption on the genotype used here.
        initializer = ealib.CategoricalProbabilisticallyCompleteInitializer(
            [[1.0 - init_p, init_p] if a == 2 else [1.0, 1.0, 1.0]
            for a in problem_spec["alphabet_size"]]
        )
        # Adaptive steering doesn't do anything for random search, hence keep archive as-is.
        archive = ealib.BruteforceArchive(objective_indices)

        return scheduler, ealib.DistributedRandomSearch(
            scheduler,
            # approach_spec["population_size"], # -- Isn't actually part of random search.
            initializer,
            archive,
            # note - actually batch size.
            # only designates how many samples are sampled at once.
            approach_spec["population_size"],
        ), noop
