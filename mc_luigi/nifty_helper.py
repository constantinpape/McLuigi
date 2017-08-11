from __future__ import division, print_function

import time

# import the proper nifty version
try:
    import nifty
    ilp_backend = 'cplex'
except ImportError:
    try:
        import nifty_with_cplex as nifty
        ilp_backend = 'cplex'
    except ImportError:
        import nifty_with_gurobi as nifty
        ilp_backend = 'gurobi'


# TODO add LP_MP
# TODO add cgc ?!
def available_factorys():
    available_base = ['greedy', 'kl']
    if nifty.Configuration.WITH_CPLEX or nifty.Configuration.WITH_GUROBI:
        available_base.append('ilp')
    available = ['fm-%s' % fctry for fctry in available_base]
    available.extend(available_base)
    return available


def run_nifty_solver(
    obj,
    factory,
    verbose=True,
    time_limit=float('inf'),
    visit_nth=1
):
    solver = factory.create(obj)
    visitor = obj.loggingVisitor(
        visitNth=visit_nth,
        verbose=verbose,
        timeLimitSolver=time_limit
    )

    ret = solver.optimize(visitor)
    energies = visitor.energies()
    runtimes = visitor.runtimes()

    return ret, energies, runtimes


def nifty_greedy_factory(
    obj,
    use_andres=False
):
    if use_andres:
        return obj.greedyAdditiveFactory()
    else:
        return obj.multicutAndresGreedyAdditiveFactory()


def nifty_fusion_move_factory(
    obj,
    backend_factory,
    n_threads=20,
    seed_fraction=0.001,
    greedy_chain=True,
    kl_chain=True,
    number_of_iterations=2000,
    n_stop=20,
    pgen_type='ws',
    parallel_per_thread=2,
    n_fuse=2,
    sigma=10
):

    assert pgen_type in ('ws', 'greedy')
    if pgen_type == 'ws':
        pgen = obj.watershedProposals(sigma=sigma, seedFraction=seed_fraction)
    else:
        pgen = obj.greedyAdditiveProposals(sigma=sigma)

    fm_factory = obj.fusionMoveBasedFactory(
        fusionMove=obj.fusionMoveSettings(mcFactory=backend_factory),
        proposalGen=pgen,
        numberOfIterations=number_of_iterations,
        numberOfParallelProposals=parallel_per_thread * n_threads,
        numberOfThreads=n_threads,
        stopIfNoImprovement=n_stop,
        fuseN=n_fuse
    )

    if kl_chain and greedy_chain:
        kl_factory = nifty_kl_factory(obj, True)
        return obj.chainedSolversFactory([kl_factory, fm_factory])
    elif kl_chain and not greedy_chain:
        kl_factory = nifty_kl_factory(obj, False)
        return obj.chainedSolversFactory([kl_factory, fm_factory])
    elif greedy_chain and not kl_chain:
        greedy = nifty_greedy_factory(obj)  # andres = True
        return obj.chainedSolversFactory([greedy, fm_factory])
    else:
        return fm_factory


def nifty_ilp_factory(obj):
    factory = obj.multicutIlpFactory(
        ilpSolver=ilp_backend,
        addThreeCyclesConstraints=True,
        addOnlyViolatedThreeCyclesConstraints=True
    )
    return factory


# TODO
def nifty_decomposer_factory(
    obj,
    backend_factory
):
    pass


# TODO params
def nifty_cgc_factory(
    obj,
    greedy_chain=True,
    kl_chain=False,
    cut_phase=False
):
    factory = obj.cgcFactory(doCutPhase=cut_phase)
    if kl_chain and greedy_chain:
        kl_factory = nifty_kl_factory(obj, True)
        return obj.chainedSolversFactory([kl_factory, factory])
    elif kl_chain and not greedy_chain:
        kl_factory = nifty_kl_factory(obj, False)
        return obj.chainedSolversFactory([kl_factory, factory])
    elif greedy_chain and not kl_chain:
        greedy = nifty_greedy_factory(obj)  # andres = True
        return obj.chainedSolversFactory([greedy, factory])
    else:
        return factory


# TODO use nifty factory once the new nifty version is properly fixed
def nifty_kl_factory(
    obj,
    greedy_chain=True,
    use_andres=True
):
    if use_andres:
        factory = obj.multicutAndresKernighanLinFactory(greedyWarmstart=greedy_chain)
    else:
        factory = obj.kernighanLinFactory(warmStartGreedy=greedy_chain)
    return factory


# TODO more mp settings
def nifty_mp_factory(
    obj,
    backend_factory=None,  # default is none which uses KL
    number_of_iterations=1000,
    timeout=0,
    n_threads=1,
    tighten=True,
    standardReparametrization='anisotropic',
    tightenReparametrization='damped_uniform',
    roundingReparametrization='damped_uniform',
    tightenIteration=2,
    tightenInterval=49,
    tightenSlope=0.1,
    tightenConstraintsPercentage=0.05,
    primalComputationInterval=13,
):

    factory = obj.multicutMpFactory(
        mcFactory=backend_factory,
        timeout=timeout,
        numberOfIterations=number_of_iterations,
        numberOfThreads=n_threads,
        tighten=tighten,
        standardReparametrization=standardReparametrization,
        tightenReparametrization=tightenReparametrization,
        roundingReparametrization=roundingReparametrization,
        tightenIteration=tightenIteration,
        tightenInterval=tightenInterval,
        tightenSlope=tightenSlope,
        tightenConstraintsPercentage=tightenConstraintsPercentage,
        primalComputationInterval=primalComputationInterval
    )
    return factory


def string_to_factory(obj, solver_type, solver_kwargs={}, backend_kwargs={}):
    if solver_type == 'greedy':
        return nifty_greedy_factory(obj)
    elif solver_type == 'kl':
        return nifty_kl_factory(obj)
    elif solver_type == 'ilp':
        return nifty_ilp_factory(obj)
    elif solver_type == 'fm-greedy':
        return nifty_fusion_move_factory(
            obj,
            backend_factory=nifty_greedy_factory(obj),
            **solver_kwargs
        )
    elif solver_type == 'fm-kl':
        return nifty_fusion_move_factory(
            obj,
            backend_factory=nifty_kl_factory(obj),
            **solver_kwargs
        )
    elif solver_type == 'fm-ilp':
        return nifty_fusion_move_factory(
            obj,
            backend_factory=nifty_ilp_factory(obj),
            **solver_kwargs
        )
    else:
        raise RuntimeError("Invalid solver_type: %s" % solver_type)


# TODO TODO TODO

def nifty_lmc_objective(
    uv_ids,
    lifted_uv_ids,
    costs,
    lifted_costs
):
    pass


def run_nifty_lmc(
    objective,
    factory
):
    pass


def nifty_lmc_fm_factory(
    objective,
    backend_factory,
    warmstart=True
):
    pass


def nifty_lmc_greedy_factory(objective):
    pass


def nifty_lmc_kl_factory(objective, warmstart=True):
    pass


def nifty_lmc_mp_factory(objective):
    pass
