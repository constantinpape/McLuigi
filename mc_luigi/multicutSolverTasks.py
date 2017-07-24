# Multicut Pipeline implemented with luigi
# Multicut Solver Tasks

import luigi

import os

from customTargets import HDF5DataTarget
from pipelineParameter import PipelineParameter
from tools import config_logger, run_decorator
from nifty_helper import run_nifty_solver, nifty_ilp_factory, string_to_factory

import logging

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


# init the workflow logger
workflow_logger = logging.getLogger(__name__)
config_logger(workflow_logger)


class McSolverFusionMoves(luigi.Task):

    problem = luigi.TaskParameter()

    def requires(self):
        return self.problem

    @run_decorator
    def run(self):

        mc_problem = self.input()

        g = nifty.graph.UndirectedGraph()

        edge_costs = mc_problem.read("costs")
        g.deserialize(mc_problem.read("graph"))

        assert g.numberOfEdges == edge_costs.shape[0]

        obj = nifty.graph.multicut.multicutObjective(g, edge_costs)
        workflow_logger.info(
            "McSolverFusionMoves: solving multicut problem with %i number of variables" % g.numberOfNodes
        )

        solver_type = PipelineParameter().globalSolverType
        if solver_type in ('fm-ilp', 'fm-kl'):
            solver_params  = dict(
                sigma=PipelineParameter().multicutSigmaFusion,
                number_of_iterations=PipelineParameter().multicutNumIt,
                n_stop=PipelineParameter().multicutNumItStopGlobal,
                n_threads=PipelineParameter().multicutNThreadsGlobal,
                n_fuse=PipelineParameter().multicutNumFuse,
                seed_fraction=PipelineParameter().multicutSeedFractionGlobal
            )

        else:
            solver_params = dict()

        workflow_logger.info("McSolverFusionMoves: Solving problems with solver %s" % solver_type)
        workflow_logger.info(
            "McSolverFusionMoves: With Params %s" % ' '.join(
                ['%s, %s,' % (str(k), str(v)) for k, v in solver_params.iteritems()]
            )
        )

        factory = string_to_factory(obj, solver_type, solver_params)
        ret, mc_energy, t_inf = run_nifty_solver(obj, factory, verbose=True)
        mc_energy = mc_energy[-1]
        t_inf = t_inf[-1]

        workflow_logger.info("McSolverFusionMoves: inference with fusion move solver in %i s" % t_inf)
        workflow_logger.info("McSolverFusionMoves: energy of the solution %f" % mc_energy)

        self.output().write(ret)

    def output(self):
        save_path = os.path.join(
            PipelineParameter().cache,
            "McSolverFusionMoves_%s.h5"
            % ("modified" if PipelineParameter().defectPipeline else "standard",)
        )
        return HDF5DataTarget(save_path)


class McSolverExact(luigi.Task):

    problem = luigi.TaskParameter()

    def requires(self):
        return self.problem

    @run_decorator
    def run(self):
        mcProblem = self.input()

        g = nifty.graph.UndirectedGraph()

        edgeCosts = mcProblem.read("costs")
        g.deserialize(mcProblem.read("graph"))

        assert g.numberOfEdges == edgeCosts.shape[0]

        obj = nifty.graph.multicut.multicutObjective(g, edgeCosts)

        factory = nifty_ilp_factory(obj)
        ret, mc_energy, t_inf = run_nifty_solver(obj, factory, verbose=True)
        mc_energy = mc_energy[-1]
        t_inf = t_inf[-1]

        workflow_logger.info("McSolverExact: inference with exact solver in %i s" % t_inf)
        workflow_logger.info("McSolverExact: energy of the solution %f" % mc_energy)

        self.output().write(ret)

    def output(self):
        save_path = os.path.join(PipelineParameter().cache, "McSolverExact.h5")
        return HDF5DataTarget(save_path)
